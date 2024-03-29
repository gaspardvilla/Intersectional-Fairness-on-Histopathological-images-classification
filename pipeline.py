from helpers.save_predictions import save_predictions
from lightning import seed_everything
from config.get_args import get_args
from helpers.init import init


# Get the arguments given by the run or default ones
args = get_args()


if __name__ == '__main__':
    
    # Unify the seed abd get the data, model and wandb module
    seed_everything(args.seed)
    data, model, wandb, trainer = init(**vars(args))
    
    # Fit the model and test it
    trainer.train_fit(model, data.TrainLoader, data.ValidationLoader, ckpt_path = args.ckpt_path)
    trainer.test(model, data.TestLoader)
    
    # Save full predictions
    if args.save_preds: save_predictions(args, data, model, wandb = wandb)