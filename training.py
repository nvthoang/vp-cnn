import torch 
import torch.nn as nn
import time
import os

#loss function
def net_loss(generated_img, target_img, lf='mse', l1:bool=False):
    assert lf in ['bce', 'mse']
    loss = nn.BCELoss() if lf=='bce' else nn.MSELoss()
    l1_loss = l1_loss(generated_img, target_img) if l1 else 0
    total_losses = loss + (100 * l1_loss)
    return total_losses


if os.path.isdir("./checkpoints")==False:
    os.makedirs("./checkpoints")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#training function
def training(model,
             train_dl,
             eval_dl=None,
             num_epochs:int=50,
             lr:float=0.0001,
             save_model_per_n_epoch:int=None,
             start_epoch:int=1,
             start_val_loss:float=100.0,
             current_best_epoch:int=1,
             loss_function='bce'):
    '''
    model: list [generator, discriminator]
    train_dl: data_loader for training set
    test_ld: data_loader for val/test set
    num_epochs: num of training epochs
    patch_dim: dim of patch returned by discriminator
    '''
    assert loss_function in ['bce', 'mse']
    cnn_net=model
    optimizer=torch.optim.Adam(cnn_net.parameters(), lr=lr)
    train_losses, test_losses = [], []
    val_total_loss=start_val_loss
    current_best_epoch=current_best_epoch
    #===========================
    #Training
    for epoch in range(start_epoch, start_epoch+num_epochs): 
        cnn_net.train()
        train_loss= 0.0
        start=time.time()
        num_batch=len(train_dl.dataset)
        for input_img, target_img in train_dl:
            optimizer.zero_grad()
            input_img=input_img.to(device)
            target_img=target_img.to(device)
            #compute loss
            output=cnn_net(input_img)
            train_loss+=net_loss(output, target_img)
            #compute gradients and run optimizer step
            train_loss.backward()
            optimizer.step()
        train_losses.append(train_loss.detach().cpu().numpy()/num_batch)
        end=time.time()
        #==========================
        if save_model_per_n_epoch!=None:
            if epoch%save_model_per_n_epoch==0:
                torch.save(cnn_net.state_dict(), f"./checkpoints/cnn_{epoch}.pth")
        #==========================
        #Save current epoch for retraining if runtime error
        torch.save(cnn_net.state_dict(), f"./checkpoints/current_cnn.pth")
        #==========================
        #Log
        log=f"Training: Epoch {epoch} - CNN net loss: {train_loss/num_batch}, Time: {end-start}"
        print(log)
        if epoch!=1:
            with open("./log.txt", "a") as f:
                f.writelines(log + "\n")
        else:
            with open("./log.txt", "w") as f:
                f.writelines(log + "\n")
        f.close()
        #===========================
        #Evaluation
        if eval_dl!=None:
            cnn_net.eval()
            with torch.no_grad():
                num_batch=len(eval_dl.dataset)
                test_loss=0.0
                for input_img, target_img in eval_dl:
                    input_img=input_img.to(device)
                    target_img=target_img.to(device)
                    #compute loss
                    test_loss+=net_loss(output, target_img)
                test_loss=test_loss.detach().cpu().numpy()/num_batch    
                test_losses.append(test_loss)
                #===========================
                #Update current best epoch and save model at the epoch
                if test_losses < val_total_loss:
                    val_total_loss = test_losses
                    current_best_epoch = epoch
                    torch.save(cnn_net.state_dict(), f"./checkpoints/best_cnn.pth")
                #===========================
                #Log
                log=f"Evaluation: Epoch {epoch} - CNN net loss: {test_loss/num_batch}"
                print(log)
                print(f"Current best epoch: {current_best_epoch}")
                with open("./log.txt", "a") as f:
                    f.writelines(log + "\n")
                f.close()
    #===========================
    if eval_dl!=None:
        return train_losses, test_losses
    else:
        return train_losses
