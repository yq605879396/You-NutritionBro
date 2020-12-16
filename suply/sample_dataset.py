import os, random, shutil

# sample images in root_dit + sub_dir, copy in to tar_dir
def moveFile(root_dir, sub_dir, tar_dir):
    if not os.path.exists(tar_dir + sub_dir):
        os.makedirs(tar_dir + sub_dir)
    pathDir = os.listdir(root_dir + sub_dir)  
    rate=0.6   # sample rate
    picknumber=int(filenumber*rate) # sample images according to the size of that directory
    sample = random.sample(pathDir, picknumber)  
    
    for name in sample:
        if name[-3:] != 'jpg':
            continue
        print (root_dir + sub_dir+"/" + name, tar_dir+sub_dir + "/"+ name)
        shutil.copy(root_dir + sub_dir+"/" + name, tar_dir+sub_dir + "/"+ name)
    return

if __name__ == '__main__':
    root_dir = "./images/" # original directory
    directories = os.listdir(root_dir) # get the directory list (each class' directory) 
    tar_dir = './sampled_images/'    # new directory
    print(os.getcwd())
    for sub_dir in directories: 
        if os.path.isdir(os.getcwd() + "/images/" + sub_dir):
            moveFile(root_dir, sub_dir, tar_dir)




















    
