import torch
from cpgan.ooppnm import img_process
import random

img_prc = img_process.Image_process()


class Z_perturb:
    def __init__(self,gen,func,target):
        '''
        gen: generation model
        func: physical simulation
        target: target value to optimize z space
        '''
        self.gen = gen.eval()
        self.func = func
        self.target = target

    def grad_cal(self,err,z1,z2,t):
        err = torch.Tensor([err])
        grad = err * (-z1*torch.sin(t)+z2*torch.cos(t))
        return torch.mean(grad)

    def grad_update(self,grad,t,eta):
        t -= eta*grad
        return t

    def compound_vec(self,z1,z2,t):
        z_n = z1*torch.cos(t) + z2*torch.sin(t)
        return z_n

    def forward_img(self,z_n,gen):
        gen.eval()
        img = gen(z_n)
        return img
    
    def clean_img(self,tensor_img):
        tensor_img = tensor_img.detach().cpu()
        process_img = tensor_img.numpy()
        return process_img

    def optimize(self,epoch,threshold,eta=0.1):
        err_list = []
        z_n = torch.randn(1,200)
        t = torch.Tensor([0.5])

        for i in range(epoch):
            z2 = torch.randn(1,200)
            z_n = self.compound_vec(z_n,z2,t)
            img = self.forward_img(z_n,self.gen)
            
            img = img_prc.clean_img(img)
            pred = self.func(img)
            if pred == None:
                continue
            err = pred - self.target
            err_list.append(abs(err))
            print(f'Epoch {i}, error: {err}')
            if abs(err) < threshold:
                return z_n,err_list,pred,i
            else:
                grad_t = self.grad_cal(err,z_n,z2,t)
                t = self.grad_update(grad_t,t,eta)
                z_n = self.compound_vec(z_n,z2,t)
        
        if abs(err)<threshold:
            return z_n,err_list,pred,i
        else:
            return None,None,None
        