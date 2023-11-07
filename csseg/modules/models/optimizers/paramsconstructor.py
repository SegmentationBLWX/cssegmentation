'''
Function:
    Implementation of ParamsConstructors
Author:
    Zhenchao Jin
'''
import copy
from ...utils import BaseModuleBuilder
from ..encoders import NormalizationBuilder


'''DefaultParamsConstructor'''
class DefaultParamsConstructor():
    def __init__(self, paramwise_cfg={}, filter_params=False, optimizer_cfg=None):
        self.paramwise_cfg = paramwise_cfg
        self.filter_params = filter_params
        self.optimizer_cfg = optimizer_cfg
    '''call'''
    def __call__(self, model):
        # fetch attributes
        paramwise_cfg, filter_params, optimizer_cfg = self.paramwise_cfg, self.filter_params, self.optimizer_cfg
        # without specific parameter rules
        if not paramwise_cfg:
            params = model.parameters() if not filter_params else filter(lambda p: p.requires_grad, model.parameters())
            return params
        # with specific parameter rules
        params = []
        self.groupparams(model, paramwise_cfg, filter_params, optimizer_cfg, params)
        return params
    '''groupparams'''
    def groupparams(self, model, paramwise_cfg, filter_params, optimizer_cfg, params, prefix=''):
        # fetch base_setting
        optimizer_cfg = copy.deepcopy(optimizer_cfg)
        if 'base_setting' in optimizer_cfg:
            base_setting = optimizer_cfg.pop('base_setting')
        else:
            base_setting = {
                'bias_lr_multiplier': 1.0, 'bias_wd_multiplier': 1.0, 'norm_wd_multiplier': 1.0,
                'lr_multiplier': 1.0, 'wd_multiplier': 1.0
            }
        # iter to group current parameters
        sorted_rule_keys = sorted(sorted(paramwise_cfg.keys()), key=len, reverse=True)
        for name, param in model.named_parameters(recurse=False):
            param_group = {'params': [param]}
            # --if `parameter requires gradient` is False
            if not param.requires_grad:
                if not filter_params:
                    params.append(param_group)
                continue
            # --find parameters with specific rules
            set_base_setting = True
            for rule_key in sorted_rule_keys:
                if rule_key not in f'{prefix}.{name}': continue
                set_base_setting = False
                param_group['lr'] = paramwise_cfg[rule_key].get('lr_multiplier', 1.0) * optimizer_cfg['lr']
                param_group['name'] = f'{prefix}.{name}' if prefix else name
                param_group['rule_key'] = rule_key
                if 'weight_decay' in optimizer_cfg:
                    param_group['weight_decay'] = paramwise_cfg[rule_key].get('wd_multiplier', 1.0) * optimizer_cfg['weight_decay']
                for k, v in paramwise_cfg[rule_key].items():
                    param_group[k] = v
                params.append(param_group)
                break
            if not set_base_setting: continue
            # --set base setting
            param_group['lr'] = optimizer_cfg['lr']
            param_group['name'] = f'{prefix}.{name}' if prefix else name
            param_group['rule_key'] = 'base_setting'
            if name == 'bias' and (not NormalizationBuilder.isnorm(model)):
                param_group['lr'] = param_group['lr'] * base_setting.get('bias_lr_multiplier', 1.0)
            else:
                param_group['lr'] = param_group['lr'] * base_setting.get('lr_multiplier', 1.0)
            if 'weight_decay' in optimizer_cfg:
                param_group['weight_decay'] = optimizer_cfg['weight_decay']
                if NormalizationBuilder.isnorm(model):
                    param_group['weight_decay'] = param_group['weight_decay'] * base_setting.get('norm_wd_multiplier', 1.0)
                elif name == 'bias':
                    param_group['weight_decay'] = param_group['weight_decay'] * base_setting.get('bias_wd_multiplier', 1.0)
                else:
                    param_group['weight_decay'] = param_group['weight_decay'] * base_setting.get('wd_multiplier', 1.0)
            params.append(param_group)
        # iter to group children parameters
        for child_name, child_model in model.named_children():
            if prefix:
                child_prefix = f'{prefix}.{child_name}'
            else:
                child_prefix = child_name
            self.groupparams(child_model, paramwise_cfg, filter_params, optimizer_cfg, params, prefix=child_prefix)
    '''isin'''
    def isin(self, param_group, param_group_list):
        param = set(param_group['params'])
        param_set = set()
        for group in param_group_list:
            param_set.update(set(group['params']))
        return not param.isdisjoint(param_set)


'''ParamsConstructorBuilder'''
class ParamsConstructorBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'DefaultParamsConstructor': DefaultParamsConstructor, 
    }
    '''build'''
    def build(self, paramwise_cfg={}, filter_params=False, optimizer_cfg={}):
        constructor_type = paramwise_cfg.pop('type', 'DefaultParamsConstructor')
        module_cfg = {
            'paramwise_cfg': paramwise_cfg, 'filter_params': filter_params, 'optimizer_cfg': optimizer_cfg, 'type': constructor_type
        }
        return super().build(module_cfg)


'''BuildParamsConstructor'''
BuildParamsConstructor = ParamsConstructorBuilder().build