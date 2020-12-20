import glob
import PIL
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import transforms as T
from imgAugmentation import *

class ZSSRdataset(Dataset):
    def __init__(self, src_img, s_factor):
        super(ZSSRdataset, self).__init__()
        self.s_factor = s_factor
        self.src_img = src_img
        im_height = self.src_img.size[0]
        im_width = self.src_img.size[1]
        
        zoom_factors = []
        for i in range(im_height // 5, im_height + 1):
            downsampled_height = i
            zoom = float(downsampled_height / im_height)
            zoom_factors.append(zoom)
                
        hres_list = []
        for zoom in zoom_factors:
            hres = self.src_img.resize((int(self.src_img.size[0] * zoom), 
                                        int(self.src_img.size[1] * zoom)), 
                                       resample=Image.BICUBIC)
            hres_list.append(hres)
            
        self.hres_list = hres_list
        
    def __getitem__(self, index):
        return self.hres_list[index]
    
    def __len__(self):
        return len(self.hres_list)
    
    @classmethod
    def from_image(cls, img, s_factor):
        return ZSSRdataset(img, s_factor)
    
    def concat(self, dataset):
        self.hres_list += dataset.hres_list
        return self
    
class ZSSRsampler(Sampler):
    def __init__(self, dataset):
        super(ZSSRsampler, self).__init__(dataset)
        self.dataset = dataset
        sizes = np.float32([(hres.size[0]*hres.size[1] / float(
            self.dataset.src_img.size[0]*self.dataset.src_img.size[1])) for hres in self.dataset.hres_list])
        self.pair_probabilities = sizes / np.sum(sizes)
        
    def __iter__(self):
        while True:
            yield random.choices(self.dataset, weights=self.pair_probabilities, k=1)[0]
