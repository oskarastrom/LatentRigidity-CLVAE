##################################################
# Imports
##################################################

import json
import numpy as np
import pytorch_lightning as pl

# Custom
from config import parse_args
from dataloader import get_dataloaders
from models import cvaecaposr
from utils import get_logger, get_callbacks, generate_save_name


# Main function
def main(args):

    # Dataloaders
    dls, data_info = get_dataloaders(args)

    # Model
    model = cvaecaposr.get_model(args, data_info)
    print("------------- known classes: ", args.known_classes)

    # Callbacks and logger
    callbacks = get_callbacks(args)
    tb_logger = get_logger(args)

    # Trainer
    if args.mode in ['train', 'training']:
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator="gpu",
            devices=1,
            callbacks=callbacks,
            num_sanity_val_steps=0,
            logger=tb_logger,
        )
        
        # Fit
        trainer.fit(
            model, 
            train_dataloaders=dls['known']['train_aug'], 
            val_dataloaders=dls['known']['validation'],
        )

        if np.prod(model.t_mean_learned.weight.shape) > 0: print("learned", model.t_mean_learned.weight[0,0].item(), model.t_mean_learned.weight.shape)
        if np.prod(model.t_mean_fixed.weight.shape) > 0: print("fixed", model.t_mean_fixed.weight[0,0].item(), model.t_mean_fixed.weight.shape)
        if np.prod(model.t_var.weight.shape) > 0: print("var", model.t_var.weight[0,0].item(), model.t_var.weight.shape)
        
        if len(args.save_name) > 0: 
            save_path = generate_save_name("checkpoints/" + args.dataset + "/" + str(args.split_num), args.save_name + "_last")
            trainer.save_checkpoint(save_path)
            
        test_out = trainer.test(model=None, dataloaders=dls['test'], ckpt_path="best")
        
        if len(args.save_name) > 0: 
            save_path = generate_save_name("checkpoints/" + args.dataset + "/" + str(args.split_num), args.save_name + "_best")
            trainer.save_checkpoint(save_path)
        return test_out
        
    elif args.mode in ['test', 'testing']:
        trainer = pl.Trainer( 
            accelerator="gpu",
            devices=1,
            callbacks=callbacks,
            logger=tb_logger,
        )

        if np.prod(model.t_mean_learned.weight.shape) > 0: print("learned", model.t_mean_learned.weight[0,0].item(), model.t_mean_learned.weight.shape)
        if np.prod(model.t_mean_fixed.weight.shape) > 0: print("fixed", model.t_mean_fixed.weight[0,0].item(), model.t_mean_fixed.weight.shape)
        if np.prod(model.t_var.weight.shape) > 0: print("var", model.t_var.weight[0,0].item(), model.t_var.weight.shape)
        
        # Test
        return trainer.test(model=model, dataloaders=dls['test'])
    else:
        raise Exception(f'Error. Mode "{args.mode}" is not supported.')



##################################################
# Main
##################################################

if __name__ == '__main__':

    # Parse args
    args = parse_args()
    print(json.dumps(vars(args), indent=4))

    # Main
    main(args)


