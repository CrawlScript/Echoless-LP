





import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from itertools import chain
import torch.nn.functional as F
from tqdm import tqdm
import time

from echoless_lp.utils.metrics_utils import LogitsBasedMetric



class TorchTrainModel(nn.Module):
    def __init__(self, metrics_dict=None, learning_rate=None, scheduler_gamma=None) -> None:

        super().__init__()

        self.metrics_dict = metrics_dict
        self.learning_rate = learning_rate
        self.scheduler_gamma = scheduler_gamma
        self.stop_training = False 

        use_float16 = False

        if use_float16:
            self.autocast_dtype = torch.float16
            self.scalar = torch.cuda.amp.GradScaler()
        else:
            self.autocast_dtype = torch.float32
            self.scalar = None

        self.optimizer = None

        self._epochs = None

 
    def predict(self, data_loader, training=False):
        last_status = self.training
        if training:
            self.train()
        else:
            self.eval()

        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=self.autocast_dtype):
                batch_y_pred_list = []
                for step, (batch_x, batch_y) in enumerate(tqdm(data_loader)):
                    batch_logits = self(batch_x)
                    batch_y_pred = self.output_activation_func(batch_logits)
                    
                    
                    
                    

                    
                    
                    batch_y_pred_list.append(batch_y_pred.cpu())

        

        y_pred = torch.concat(batch_y_pred_list, dim=0)

        self.train(last_status)
        return y_pred
    


    def evaluate(self, data_loader, log_prefix):
        self.eval()
        
        
        
        

        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=self.autocast_dtype):
                batch_y_pred_list = []
                batch_y_list = []
                losses_list = []
                for step, (batch_x, batch_y) in enumerate(tqdm(data_loader)):
                    batch_logits = self(batch_x)
                    
                    batch_losses = self.loss_func(batch_logits, batch_y)
                    
                    if self.multi_label:
                        batch_y_pred = (torch.sigmoid(batch_logits) > 0.5).float()
                    else:
                        batch_y_pred = torch.argmax(batch_logits, dim=-1)

                    if self.metrics_dict is not None:
                        for metric in self.metrics_dict.values():
                            if isinstance(metric, LogitsBasedMetric):
                                metric(batch_logits, batch_y)
                            else:
                                metric(batch_y_pred, batch_y)

                    losses_list.append(batch_losses.detach().cpu().numpy())
                    batch_y_pred_list.append(batch_y_pred.detach().cpu().numpy())
                    batch_y_list.append(batch_y.detach().cpu().numpy())

        losses = np.concatenate(losses_list, axis=0)
        loss = losses.mean()

        
        


        

        
        
        

        
        logs = {}

        logs["{}_loss".format(log_prefix)] = loss
        

        
        
        

        if self.metrics_dict is not None:
            with torch.no_grad():        
                for metric_name, metric in self.metrics_dict.items():
                    logs["{}_{}".format(log_prefix, metric_name)] = metric.compute().item()
                    metric.reset()

        return logs




    def train_step(self, batch_data):
        return {}


  
    def train_epoch(self, epoch, train_data_loader):
        self.train()

        batch_results_dict = {}
        step_pbar = tqdm(train_data_loader)
        for step, batch_data in enumerate(step_pbar):    
            batch_result = self.train_step(batch_data)
            with torch.no_grad():
                for key, value in batch_result.items():
                    if key not in batch_results_dict:
                        batch_results_dict[key] = []
                    batch_results_dict[key].append(value)
    
            step_pbar.set_postfix(
                {key: "{:.4f}".format(value.item()) for key, value in batch_result.items()}
            )


        if self.scheduler is not None:
            self.scheduler.step()
            print("current learning_rate: ", self.scheduler.get_last_lr())

        with torch.no_grad():
            logs = {
                key: torch.stack(value, dim=0).mean().item() for key, value in batch_results_dict.items() 
            }

        return logs




    def fit(self, train_data, 
            epochs, 
            validation_data, 
            validation_freq, 
            callbacks=None,
            initial_epoch=0,
            ):
        
        self._epochs = epochs

        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            print("create optimizer ...")

            if self.scheduler_gamma is not None:
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=self.scheduler_gamma)
            else:
                self.scheduler = None

        if callbacks is None:
            callbacks = []
        
        for callback in callbacks:
            callback.model = self

        for callback in callbacks:
            callback.on_train_begin()
            
        for epoch in range(initial_epoch, epochs):
            logs = {"epoch": epoch}
            self.train()
            print("start epoch {}:".format(epoch))
            train_logs = self.train_epoch(epoch, train_data)
            
            logs = {
                **logs,
                **train_logs
            }

            
            
            if (epoch + 1) % validation_freq == 0:
                self.eval()
                eval_start_time = time.time()
                validation_logs = self.evaluate(validation_data, log_prefix="val")
                logs = {
                    **logs,
                    **validation_logs
                }
                print("==== eval_time: ", time.time() - eval_start_time)


            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)
            
            
            if (epoch + 1) % validation_freq == 0:
                
                print("epoch = {}\tlogs = {}".format(epoch, logs))

            if self.stop_training:
                print("early stop ...")
                break






class CommonTorchTrainModel(TorchTrainModel):
    def __init__(self, metrics_dict=None, multi_label=False, loss_func=None, learning_rate=None, scheduler_gamma=None, train_strategy="common", num_views=None, cl_rate=None) -> None:

        super().__init__(metrics_dict, learning_rate, scheduler_gamma)

        self.multi_label = multi_label
        self.train_strategy = train_strategy
        self.num_views = num_views
        self.cl_rate = cl_rate
        self.device = "cuda"

        if loss_func is not None:
            self.loss_func = loss_func
        else:
            if self.multi_label:
                self.loss_func = torch.nn.BCEWithLogitsLoss(reduction="none")
                self.output_activation_func = torch.nn.Sigmoid()
            else:
                self.loss_func = torch.nn.CrossEntropyLoss(reduction="none")
                self.output_activation_func = torch.nn.Softmax(dim=-1)

    

        def weighted_cross_entropy(logits, labels):
            probs = F.softmax(logits, dim=-1)
            probs = probs[torch.arange(0, probs.size(0)), labels]
            probs = probs.detach()

            weights = torch.ones_like(probs)
            weights[probs > 0.8] = 0.0
            

            scale = torch.tensor(weights.size(0)).float() / (weights.sum() + 1e-8)
            weights *= scale

            losses = self.loss_func(logits, labels)
            loss = (losses * weights).mean()
            return loss


        self.optimizer = None


    def compute_kl_loss(self, logits, batch_x):
        batch_label_x = batch_x[-self.num_class_groups:]
        pseudo_label_list = [torch.stack(h_list, dim=0).mean(dim=0) for h_list in batch_label_x]
        pseudo_label = torch.stack(pseudo_label_list, dim=0).mean(dim=0)

        kl_loss = self.loss_func(logits, pseudo_label).mean()
        return kl_loss
    
    def compute_l2_loss(self):
        l2_loss = 0.0
        for name, param in self.named_parameters():
            if "weight" in name:
                l2_loss += (param ** 2).sum() * 0.5
        print("l2_loss = {}".format(l2_loss.item()))
        return l2_loss * 1e-5

    
    def common_forward_and_compute_loss(self, batch_x, batch_y):

        logits = self(batch_x)
        losses = self.loss_func(logits, batch_y)
        loss = losses.mean()

        
        

        
        
        

        

        return logits, loss

    def cl_forward_and_compute_loss(self, batch_x, batch_y, batch_train_mask):

                
        ce_loss_list = []
        y_pred_list = []

        logits_list = [self(batch_x) for _ in range(self.num_views)]
        ce_loss_list = [self.loss_func(logits[batch_train_mask], batch_y[batch_train_mask]).mean() 
                        for logits in logits_list]
        
        
        
        
        ce_loss = torch.stack(ce_loss_list, dim=0).sum(dim=0)


        y_pred_list = [self.output_activation_func(logits) for logits in logits_list]

        
        
        
        

        stacked_y_preds = torch.stack(y_pred_list, dim=1)
        mean_y_pred = stacked_y_preds.mean(dim=1)

        pseudo_y = torch.argmax(mean_y_pred, dim=-1)

        
        
        
        
        
        
        
        
        

        
        cl_loss_list = [self.loss_func(logits, pseudo_y).mean() 
                                for logits in logits_list]
        cl_loss = torch.stack(cl_loss_list, dim=0).sum(dim=0)

        
        loss = ce_loss + cl_loss * self.cl_rate


        
        
        
        

        
            
        return logits_list[0], loss

    def train_step(self, batch_data):

        
        self.train()
        with torch.autocast(device_type=self.device, dtype=self.autocast_dtype):

            if self.train_strategy == "common":
                batch_x, batch_y = batch_data
                logits, loss = self.common_forward_and_compute_loss(batch_x, batch_y)

            elif self.train_strategy == "cl":
                batch_x, batch_y, batch_train_mask = batch_data
                logits, loss = self.cl_forward_and_compute_loss(batch_x, batch_y, batch_train_mask)
                        
            elif self.train_strategy == "cl_conf":
                batch_x, batch_y, batch_train_mask = batch_data
                logits, loss = self.cl_conf_forward_and_compute_loss(batch_x, batch_y, batch_train_mask)

            elif self.train_strategy == "cl_cos":
                batch_x, batch_y, batch_train_mask = batch_data
                logits, loss = self.cl_cos_forward_and_compute_loss(batch_x, batch_y, batch_train_mask)

            elif self.train_strategy == "cl_soft":
                batch_x, batch_y, batch_train_mask = batch_data
                logits, loss = self.cl_soft_forward_and_compute_loss(batch_x, batch_y, batch_train_mask)
            elif self.train_strategy == "cl_weighted":
                batch_x, batch_y, batch_train_mask, weights = batch_data
                logits, loss = self.cl_weighted_forward_and_compute_loss(batch_x, batch_y, batch_train_mask, weights)

            else:
                raise Exception("not supported yet")
            
        

        self.optimizer.zero_grad()
        if self.scalar is None:
            loss.backward()
            self.optimizer.step()        
        else:
            self.scalar.scale(loss).backward()
            self.scalar.step(self.optimizer)
            self.scalar.update()

        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=self.autocast_dtype):
                if self.multi_label:
                    batch_y_pred = logits > 0.0
                else:
                    batch_y_pred = logits.argmax(dim=-1)
                    
                batch_corrects = (batch_y_pred == batch_y).float()
                batch_accuracy = batch_corrects.mean()

        return {
            "loss": loss, 
            "accuracy": batch_accuracy
        }
