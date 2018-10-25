
# [Dog Breed Web App](https://github.com/govinsprabhu/dog-breed-webapp/) 

## About the project

  This web app is used to **detect the Dog breed using the CNN**. You can upload human image also. If it is human, the app will identify the closest dog breed that resembles the face.   
  The front end is made with Angular 5 and backend is made with python Flask and Keras. This repository contains front end and back end code.

## Front-end

 The front end is made with **Angular 5**. For running, go to the Dog-breed-UI directory inside the front-end folder, run following command
 
```nodejs
npm-install
ng serve
```
Go to the URL **localhost:4200**

## Back-end
 
 The back-end code is written in **Python**. **Flask** is the server framework used. **Keras** is used to code the machine learning model. **Xception** model is used for CNN classification. You can run the application using the following command.

```python
python Dog-breed.py
```

## Sample
 If you added an image of Dog, the output will be look like
![alt text](https://github.com/govinsprabhu/dog-breed-webapp/blob/master/images/dog.png)
 
 Else you added the image of Human
 ![alt text](https://github.com/govinsprabhu/dog-breed-webapp/blob/master/images/human.png)
