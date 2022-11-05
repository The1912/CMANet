import os
import torch

device = 'cuda'
sav_dir = 'savefile'
def load_checkpoint(model, optimizer, scheduler, model_file):
    if os.path.isfile(model_file):
        print("=> loading checkpoint '{}'".format(model_file))
        checkpoint = torch.load(model_file, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        if device == 'cuda':
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_file, checkpoint['epoch']))
        epoch = checkpoint['epoch']
        return epoch + 1
    else:
        print("=> no checkpoint found at '{}'".format(model_file))
        os._exit(0)



def save_checkpoint(model, optimizer, scheduler, epoch):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    ckpt_model_filename = os.path.join(sav_dir, "ckpt_epoch_{:0.2f}.pth".format(epoch))
    path = os.path.join(ckpt_model_filename)
    torch.save(state, path)
    print('{:>2} has been successfully saved'.format(path))