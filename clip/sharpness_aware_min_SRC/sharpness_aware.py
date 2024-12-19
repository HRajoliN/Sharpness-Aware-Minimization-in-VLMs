import torch
from .utils import enable_running_stats, disable_running_stats
import contextlib
from torch.distributed import ReduceOp
from collections import defaultdict
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

        

#--------------------------- SAM ------------------------------
#--------------------------------------------------------------
class SAM_optim(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, model, rho_scheduler, grad_scaler=None, grad_reduce='mean', cfg=None, **kwargs):
        defaults = dict(adaptive=cfg.TRAINER.SAMPLe.ADAPTIVE, **kwargs)
        super().__init__(params, defaults)
        self.model = model
        self.base_optimizer = base_optimizer
        self.rho_scheduler = rho_scheduler
        self.param_groups = self.base_optimizer.param_groups
        self.adaptive = cfg.TRAINER.SAMPLe.ADAPTIVE
        # self.perturb_eps = cfg.TRAINER.SAMPLe.RHO
        self.alpha = cfg.TRAINER.SAMPLe.ALPHA
        self.lambda_EMA = cfg.TRAINER.SAMPLe.EMA_LAMBDA
        self.normalization = cfg.TRAINER.SAMPLe.NORMALIZATION
        self.scaler = grad_scaler
        self.prec = cfg.TRAINER.SAMPLe.PREC
        self.IMAGE_LOSS_WEIGHT = cfg.TRAINER.PROMPTSRC.IMAGE_LOSS_WEIGHT
        self.TEXT_LOSS_WEIGHT = cfg.TRAINER.PROMPTSRC.TEXT_LOSS_WEIGHT

        # initialize self.rho_t
        self.update_rho_t()
        
        # set up reduction for gradient across workers
        if grad_reduce.lower() == 'mean':
            if hasattr(ReduceOp, 'AVG'):
                self.grad_reduce = ReduceOp.AVG
                self.manual_average = False
            else: # PyTorch <= 1.11.0 does not have AVG, need to manually average across processes
                self.grad_reduce = ReduceOp.SUM
                self.manual_average = True
        elif grad_reduce.lower() == 'sum':
            self.grad_reduce = ReduceOp.SUM
            self.manual_average = False
        else:
            raise ValueError('"grad_reduce" should be one of ["mean", "sum"].')
    
    @torch.no_grad()
    def update_rho_t(self):
        self.rho_t = self.rho_scheduler.step()
        return self.rho_t

    # def _dot_product(self, tensor1, tensor2):
    #     # Ensure the tensors are flattened and have the same shape
    #     tensor1_flat = tensor1.view(-1)
    #     tensor2_flat = tensor2.view(-1)

    #     if tensor1_flat.size() != tensor2_flat.size():
    #         raise ValueError("Tensors must have the same number of elements for dot product")

    #     # Compute the dot product
    #     return torch.dot(tensor1_flat, tensor2_flat)

    @torch.no_grad()
    def fg_update_and_SAM_nonSAM_pert(self, rho):
        
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            if param.requires_grad:  # Only learnable parameters (prompts) will pass
                # Get the past full-gradient average or initialize it
                if "SAM_pert" not in self.state[param]:
                    grad_norm = self._grad_norm( weight_adaptive = self.adaptive )
                    self.state[param]["SAM_pert"] = rho / grad_norm * param.grad
                else:
                    grad_norm = self._grad_norm( weight_adaptive = self.adaptive )
                    self.state[param]["SAM_pert"] = rho / grad_norm * param.grad
                    

    
    @torch.no_grad()
    def perturb_weights(self):
        for group in self.param_groups:            
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_g"] = p.grad.data.clone()
                e_w = self.state[p]["SAM_pert"].to(p)
                if self.adaptive:
                    e_w *= torch.pow(p, 2)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]['e_w'] = e_w
                
    @torch.no_grad()
    def unperturb(self):
        for group in self.param_groups:
            for p in group['params']:
                if 'e_w' in self.state[p].keys():
                    p.data.sub_(self.state[p]['e_w'])

    @torch.no_grad()
    def gradient_decompose(self, alpha=0.0):

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                sam_grad = self.state[p]['old_g'] * 0.5 - p.grad * 0.5
                p.grad.data.add_(sam_grad)

    @torch.no_grad()
    def _sync_grad(self):
        if torch.distributed.is_initialized(): # synchronize final gardients
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    if self.manual_average:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
                        world_size = torch.distributed.get_world_size()
                        p.grad.div_(float(world_size))
                    else:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
        return

    @torch.no_grad()
    def _grad_norm(self, by=None, weight_adaptive=False):
        #shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        if not by:

            norm = torch.norm(
                    torch.stack([
                        ( (torch.abs(p.data) if weight_adaptive else 1.0) * p.grad).norm(p=2)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )

        else:

            norm = torch.norm(
                torch.stack([
                    ( (torch.abs(p.data) if weight_adaptive else 1.0) * self.state[p][by]).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )

        return norm

    # def norm(tensor_list: List[torch.tensor], p=2):
    #     """Compute p-norm for tensor list"""
    #     return torch.cat([x.flatten() for x in tensor_list]).norm(p)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
        
    def maybe_no_sync(self):
        if torch.distributed.is_initialized():
            return self.model.no_sync()
        else:
            return contextlib.ExitStack()

    @torch.no_grad()
    def set_closure(self, loss_fn, inputs, targets, **kwargs):
        # create self.forward_backward_func, which is a function such that
        # self.forward_backward_func() automatically performs forward and backward passes.
        # This function does not take any arguments, and the inputs and targets data
        # should be pre-set in the definition of partial-function

        def get_grad():
            self.base_optimizer.zero_grad()
            
            if self.prec == "amp":                
                with torch.enable_grad():
                    with autocast():
                        loss_ce, normalized_text_features, zs_clip_text_embeddings, zs_image_embedd, image_ft, \
                        zero_shot_logits, logits = self.model(inputs, targets)
                        # Calculate the L_SCL_text loss
                        loss_scl_text = F.l1_loss(normalized_text_features, zs_clip_text_embeddings.cuda(),
                                                reduction='mean') * self.TEXT_LOSS_WEIGHT
                        # Calculate the L_SCL_image loss
                        loss_scl_image = F.l1_loss(image_ft, zs_image_embedd.cuda(),
                                                reduction='mean') * self.IMAGE_LOSS_WEIGHT
                        # Now calculate L_SCL_logits
                        L_SCL_logits = F.kl_div(
                            F.log_softmax(logits / 1, dim=1),
                            F.log_softmax(zero_shot_logits / 1, dim=1),
                            reduction='sum',
                            log_target=True
                        ) * (1 * 1) / logits.numel()
                        L_SCL = (L_SCL_logits + loss_scl_text + loss_scl_image)
                        loss = (loss_ce + L_SCL)
                loss_value = loss.data.clone().detach()
                self.scaler.scale(loss).backward()
            else:
                with torch.enable_grad():
                    loss_ce, normalized_text_features, zs_clip_text_embeddings, zs_image_embedd, image_ft, \
                    zero_shot_logits, logits = self.model(inputs, targets)
                    # Calculate the L_SCL_text loss
                    loss_scl_text = F.l1_loss(normalized_text_features, zs_clip_text_embeddings.cuda(),
                                            reduction='mean') * self.cfg.TRAINER.PROMPTSRC.TEXT_LOSS_WEIGHT
                    # Calculate the L_SCL_image loss
                    loss_scl_image = F.l1_loss(image_ft, zs_image_embedd.cuda(),
                                            reduction='mean') * self.cfg.TRAINER.PROMPTSRC.IMAGE_LOSS_WEIGHT
                    # Now calculate L_SCL_logits
                    L_SCL_logits = F.kl_div(
                        F.log_softmax(logits / 1, dim=1),
                        F.log_softmax(zero_shot_logits / 1, dim=1),
                        reduction='sum',
                        log_target=True
                    ) * (1 * 1) / logits.numel()
                    L_SCL = (L_SCL_logits + loss_scl_text + loss_scl_image)
                    loss = (loss_ce + L_SCL)
                loss_value = loss.data.clone().detach()
                loss.backward()
            return logits, loss_value

        self.forward_backward_func = get_grad

    @torch.no_grad()
    def step(self, closure=None):

        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func

        with self.maybe_no_sync():
            # get gradient
            _ , loss_value = get_grad()

            #compute ERA to get approximated version of fully-gradient
            self.fg_update_and_SAM_nonSAM_pert(rho=self.rho_t)
            
            # perturb weights
            self.perturb_weights()

            # disable running stats for second pass
            disable_running_stats(self.model)

            # get gradient at perturbed weights
            output , loss_value_perturb = get_grad()

            # decompose and get new update direction
            self.gradient_decompose(self.alpha)

            # unperturb
            self.unperturb()
            
        # synchronize gradients across workers
        self._sync_grad()    

        if self.prec == "amp":
            self.scaler.step(self.base_optimizer)
            self.scaler.update()
        else:
            # update with new directions
            self.base_optimizer.step()

        # enable running stats
        enable_running_stats(self.model)

        return output , loss_value_perturb


#------------------------ SAMPLe ------------------------------
#--------------------------------------------------------------
class SAMPLe_optim(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, model, rho_scheduler, grad_scaler=None, grad_reduce='mean', cfg=None, **kwargs):
        defaults = dict(adaptive=cfg.TRAINER.SAMPLe.ADAPTIVE, **kwargs)
        super().__init__(params, defaults)
        self.model = model
        self.base_optimizer = base_optimizer
        self.rho_scheduler = rho_scheduler
        self.param_groups = self.base_optimizer.param_groups
        self.adaptive = cfg.TRAINER.SAMPLe.ADAPTIVE
        # self.perturb_eps = cfg.TRAINER.SAMPLe.RHO
        self.alpha = cfg.TRAINER.SAMPLe.ALPHA
        self.lambda_EMA = cfg.TRAINER.SAMPLe.EMA_LAMBDA
        self.normalization = cfg.TRAINER.SAMPLe.NORMALIZATION
        self.scaler = grad_scaler
        self.prec = cfg.TRAINER.SAMPLe.PREC
        self.IMAGE_LOSS_WEIGHT = cfg.TRAINER.PROMPTSRC.IMAGE_LOSS_WEIGHT
        self.TEXT_LOSS_WEIGHT = cfg.TRAINER.PROMPTSRC.TEXT_LOSS_WEIGHT

        # initialize self.rho_t
        self.update_rho_t()
        
        # set up reduction for gradient across workers
        if grad_reduce.lower() == 'mean':
            if hasattr(ReduceOp, 'AVG'):
                self.grad_reduce = ReduceOp.AVG
                self.manual_average = False
            else: # PyTorch <= 1.11.0 does not have AVG, need to manually average across processes
                self.grad_reduce = ReduceOp.SUM
                self.manual_average = True
        elif grad_reduce.lower() == 'sum':
            self.grad_reduce = ReduceOp.SUM
            self.manual_average = False
        else:
            raise ValueError('"grad_reduce" should be one of ["mean", "sum"].')
    
    @torch.no_grad()
    def update_rho_t(self):
        self.rho_t = self.rho_scheduler.step()
        return self.rho_t

    # def _dot_product(self, tensor1, tensor2):
    #     # Ensure the tensors are flattened and have the same shape
    #     tensor1_flat = tensor1.view(-1)
    #     tensor2_flat = tensor2.view(-1)

    #     if tensor1_flat.size() != tensor2_flat.size():
    #         raise ValueError("Tensors must have the same number of elements for dot product")

    #     # Compute the dot product
    #     return torch.dot(tensor1_flat, tensor2_flat)

    @torch.no_grad()
    def fg_update_and_SAM_nonSAM_pert(self, rho):
        
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            
            if param.requires_grad:  # Only learnable parameters (prompts) will pass
                self.state[param]["old_g"] = torch.clone(param.grad).detach()
                # Get the past full-gradient average or initialize it
                if "full_grad" not in self.state[param]:
                    # print("full grad is not innnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn")
                    self.state[param]["full_grad"] = torch.clone(param.grad).detach()
                    self.state[param]["nonSAM_pert"] = torch.zeros_like(param.grad)
                    grad_norm = self._grad_norm( weight_adaptive = self.adaptive )
                    self.state[param]["SAM_pert"] = rho / grad_norm * param.grad
                else:
                    # print("full grad is not hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
                    # Update m_t based on the equation: m_t = lambda * m_t-1 + (1 - lambda) * gradient
                    self.state[param]["full_grad"].mul_(self.lambda_EMA).add_((1 - self.lambda_EMA) * param.grad)
                    grad_norm = self._grad_norm( weight_adaptive = self.adaptive ) + 1e-10
                    grad_norm_fg = self._grad_norm( by = "full_grad", weight_adaptive = self.adaptive ) + 1e-10   # Computes grad_norm for state[param]["full_grad"]
                    self.state[param]["SAM_pert"] = rho / grad_norm * param.grad
                    
                    if self.normalization:
                        self.state[param]["nonSAM_pert"] = self.alpha * (param.grad - F.cosine_similarity(self.state[param]["full_grad"], param.grad, dim=1)[:,None] * (1 / grad_norm_fg) * (self.state[param]["full_grad"]))
                    else:
                        self.state[param]["nonSAM_pert"] = self.alpha * (param.grad - F.cosine_similarity(self.state[param]["full_grad"], param.grad, dim=1)[:,None] * (grad_norm / grad_norm_fg) * (self.state[param]["full_grad"]))
    
    @torch.no_grad()
    def perturb_weights(self):
        grad_norm = self._grad_norm( weight_adaptive = self.adaptive )
        grad_norm_fg = self._grad_norm( by = "full_grad", weight_adaptive = self.adaptive )   # Computes grad_norm for state[param]["full_grad"]
        for group in self.param_groups:            
            for p in group["params"]:
                if p.grad is None: continue
                # self.state[p]["old_g"] = p.grad.data.clone()
                e_w = self.state[p]["SAM_pert"].to(p) - self.state[p]["nonSAM_pert"].to(p)
                if self.adaptive:
                    e_w *= torch.pow(p, 2)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]['e_w'] = e_w
                
    # @torch.no_grad()
    # def unperturb(self):
    #     for group in self.param_groups:
    #         for p in group['params']:
    #             if 'SAM_pert' in self.state[p].keys():
    #                 p.data.sub_(self.state[p]['SAM_pert'])

    @torch.no_grad()
    def unperturb(self):
        for group in self.param_groups:
            for p in group['params']:
                if 'e_w' in self.state[p].keys():
                    p.data.sub_(self.state[p]['e_w'])

    @torch.no_grad()
    def gradient_decompose(self, alpha=0.0):

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                sam_grad = self.state[p]['old_g'] * 0.5 - p.grad * 0.5
                p.grad.data.add_(sam_grad)

    @torch.no_grad()
    def _sync_grad(self):
        if torch.distributed.is_initialized(): # synchronize final gardients
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    if self.manual_average:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
                        world_size = torch.distributed.get_world_size()
                        p.grad.div_(float(world_size))
                    else:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
        return

    @torch.no_grad()
    def _grad_norm(self, by=None, weight_adaptive=False):
        #shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        if not by:

            norm = torch.norm(
                    torch.stack([
                        ( (torch.abs(p.data) if weight_adaptive else 1.0) * p.grad).norm(p=2)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )

        else:

            norm = torch.norm(
                torch.stack([
                    ( (torch.abs(p.data) if weight_adaptive else 1.0) * self.state[p][by]).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )

        return norm

    # def norm(tensor_list: List[torch.tensor], p=2):
    #     """Compute p-norm for tensor list"""
    #     return torch.cat([x.flatten() for x in tensor_list]).norm(p)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
        
    def maybe_no_sync(self):
        if torch.distributed.is_initialized():
            return self.model.no_sync()
        else:
            return contextlib.ExitStack()

    @torch.no_grad()
    def set_closure(self, loss_fn, inputs, targets, **kwargs):
        # create self.forward_backward_func, which is a function such that
        # self.forward_backward_func() automatically performs forward and backward passes.
        # This function does not take any arguments, and the inputs and targets data
        # should be pre-set in the definition of partial-function

        def get_grad():
            self.base_optimizer.zero_grad()
            
            if self.prec == "amp":                
                with torch.enable_grad():
                    with autocast():
                        loss_ce, normalized_text_features, zs_clip_text_embeddings, zs_image_embedd, image_ft, \
                        zero_shot_logits, logits = self.model(inputs, targets)
                        # Calculate the L_SCL_text loss
                        loss_scl_text = F.l1_loss(normalized_text_features, zs_clip_text_embeddings.cuda(),
                                                reduction='mean') * self.cfg.TRAINER.PROMPTSRC.TEXT_LOSS_WEIGHT
                        # Calculate the L_SCL_image loss
                        loss_scl_image = F.l1_loss(image_ft, zs_image_embedd.cuda(),
                                                reduction='mean') * self.cfg.TRAINER.PROMPTSRC.IMAGE_LOSS_WEIGHT
                        # Now calculate L_SCL_logits
                        L_SCL_logits = F.kl_div(
                            F.log_softmax(logits / 1, dim=1),
                            F.log_softmax(zero_shot_logits / 1, dim=1),
                            reduction='sum',
                            log_target=True
                        ) * (1 * 1) / logits.numel()
                        L_SCL = (L_SCL_logits + loss_scl_text + loss_scl_image)
                        loss = (loss_ce + L_SCL)
                loss_value = loss.data.clone().detach()
                self.scaler.scale(loss).backward()
            else:
                with torch.enable_grad():
                    loss_ce, normalized_text_features, zs_clip_text_embeddings, zs_image_embedd, image_ft, \
                    zero_shot_logits, logits = self.model(inputs, targets)
                    # Calculate the L_SCL_text loss
                    loss_scl_text = F.l1_loss(normalized_text_features, zs_clip_text_embeddings.cuda(),
                                            reduction='mean') * self.TEXT_LOSS_WEIGHT
                    # Calculate the L_SCL_image loss
                    loss_scl_image = F.l1_loss(image_ft, zs_image_embedd.cuda(),
                                            reduction='mean') * self.IMAGE_LOSS_WEIGHT
                    # Now calculate L_SCL_logits
                    L_SCL_logits = F.kl_div(
                        F.log_softmax(logits / 1, dim=1),
                        F.log_softmax(zero_shot_logits / 1, dim=1),
                        reduction='sum',
                        log_target=True
                    ) * (1 * 1) / logits.numel()
                    L_SCL = (L_SCL_logits + loss_scl_text + loss_scl_image)
                    loss = (loss_ce + L_SCL)
                loss_value = loss.data.clone().detach()
                loss.backward()
            return logits, loss_value

        self.forward_backward_func = get_grad

    @torch.no_grad()
    def step(self, closure=None):

        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func

        with self.maybe_no_sync():
            # get gradient
            output , loss_value = get_grad()

            #compute ERA to get approximated version of fully-gradient
            self.fg_update_and_SAM_nonSAM_pert(rho=self.rho_t)
            
            # perturb weights
            self.perturb_weights()

            # disable running stats for second pass
            disable_running_stats(self.model)

            # get gradient at perturbed weights
            _ , loss_value_perturb = get_grad()

            # decompose and get new update direction
            self.gradient_decompose(self.alpha)

            # unperturb
            self.unperturb()
            
        # synchronize gradients across workers
        self._sync_grad()    

        if self.prec == "amp":
            self.scaler.step(self.base_optimizer)
            self.scaler.update()
        else:
            # update with new directions
            self.base_optimizer.step()

        # enable running stats
        enable_running_stats(self.model)

        # return output , loss_value + loss_value_perturb
        return output , loss_value