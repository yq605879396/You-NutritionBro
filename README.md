# Your NutritionBro
Course Project for Deep Learning. Input a food image and then print out ingredient list and nutrition list.    

Mengxi Wu mw4355@nyu.edu  
Qi Yin qy652@nyu.edu    

Professor: Chinmay Hegde        

## What is it?

***

Food is essential for our lives!   
If you want to know your food better you can just take a photo of  the it and submit to your Your Nutrition Bro.
It will automatically process your image and prints out ingredients list as well as nutrition set. 

<img src="https://github.com/yq605879396/Your-NutritionBro/blob/main/images/show2.png" width="300" height="300" /> <img src="https://github.com/yq605879396/Your-NutritionBro/blob/main/images/show1.png" width="300" height="300" />
<img src="https://github.com/yq605879396/Your-NutritionBro/blob/main/images/show3.png" width="600" height="400" /> 

_credit to "ins:etn.co_mam"_

## File Structure

***

**./data:**  
Folder "data" has the text information from dataset [ingredient101](http://www.ub.edu/cvub/ingredients101/)  
You can download corresponding images [food101](https://www.kaggle.com/kmader/food41)

**./test_img:**  
This folder contains test images. You can also add your own images.     

**./nutribro_model**   
This folder contains files for model.        

**./suply**  
This folder contains codes for vocabulary builder, data loader, dataset sampler and some helper functions.  

**./root**:  
[train.py](https://github.com/yq605879396/Your-NutritionBro/blob/main/train.py): train the model  
[test.py](https://github.com/yq605879396/Your-NutritionBro/blob/main/test.py): test the model's accuracy and F1 scores <br>
[test.ipynb](https://github.com/yq605879396/Your-NutritionBro/blob/main/test.ipynb): test the model, process the image in test_img folder, and print ingredients list and nutrition list.

Result for the preprocess (can be regenerated by running suply/build_vocab): <br>
*datatest.pkl/ datatrain.pkl / dataval.pkl*: processed data (needed when training the model)  
*vocab_ingrs.pkl*: generated vocabulary (needed when training or testing the model) 

## How to use it?

***

### To train the model:  
In root directory,
```python
python train.py
```

Or you can download pretrained model here:  
[Res18 + 6 epoch](https://drive.google.com/file/d/1ycciUE9VthbnHPgRc9iLLZnPvVg2pvLO/view?usp=sharing)  
[Res50 + 6 epoch](https://drive.google.com/file/d/1S_r9XkNNpigHHtrBSD8Hy7obSQKsnzuf/view?usp=sharing)  

### To Test the model:  
To obtain accuracy and F1 scores, in root directory run: 
```python
python test.py
``` 
To obtain ingredients list and nutrition list run test.ipynb
### Reference
Mutilhead Attention used in this project is referred to Inverse Cooking [link](https://github.com/facebookresearch/inversecooking)
***

