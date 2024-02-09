from dataprocess.dataclass import Data
from dataprocess.dataloader import load_data
from config.get_args import get_args


args = get_args()


if __name__ == '__main__':
    # a, b, c, d = load_data(**vars(args))
    d = Data(**vars(args))
    
    print(0)