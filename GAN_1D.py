#First GAN with 1D
"""
Este programa muestra la primera prueba hecha de una GAN
se tomo el codigo expuesto en el libro de Jason Brownlee; Generative Adversarial
Networks with python

"""
#Importar librerias
import numpy as np
from numpy import hstack
from numpy import zeros
from numpy import ones
from numpy.random import rand
from numpy.random import randn
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

## Definimos una funcion con la que trabajaremos
f_objetivo=lambda x: x*x
x =np.linspace(-0.5,0.5,10)
fx = f_objetivo(x)
plt.plot(x,fx)
plt.show()
# Abrir archivo de texto
datos= open("datos.txt",'wt')
# define the standalone descriminator model
def define_discriminator(n_inputs=2):
    ## define model of NN
    model = Sequential()
    model.add(Dense(25, 
                    activation='relu',
                    kernel_initializer='he_uniform',
                    input_dim=n_inputs))
    model.add(Dense(1,
                    activation='sigmoid'))
    
    ## compile model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

#define the standalone generator model
def define_generator(latent_dim, n_outputs=2):
    ## define model of NN
    model =Sequential()
    model.add(Dense(15, 
                    activation='relu',
                    kernel_initializer='he_uniform',
                    input_dim=latent_dim))
    model.add(Dense(n_outputs,
                    activation='linear'))
    return model

# define the combine G and D model, for updating the generator

def define_gan(generator,discriminator):
    #make weights in the discriminator not trainable
    discriminator.trainable = False
    #connect them
    model =Sequential()
    #add generator
    model.add(generator)
    #add discriminator
    model.add(discriminator)
    #compile model 
    model.compile(loss ='binary_crossentropy',
                  optimizer ='adam')
    return model

# generate n real samples with class labels
def generate_real_samples(n):
    #generate inputs in -0.5,0.5
    X1= rand(n) -0.5
    #generate outputs x^2
    X2 = X1*X1
    #stack arrays (pilas)
    X1 = X1.reshape(n,1)
    X2 = X2.reshape(n,1)
    X = hstack((X1,X2)) # juntar X1 y X2
    #generte class lables
    y=ones((n,1))
    return X,y    

# generate point in laten space as input for the generator
def generate_latent_points(latent_dim,n):
    #generte point in the laten space
    x_input = randn(latent_dim*n)
    #reshape into a batch of inputs for the network
    x_input = x_input.reshape(n, latent_dim)
    return x_input

#use the generator to generate n fake examples
#with class labels
def generate_fake_samples(generator,latent_dim,n):
    #generte point in the laten space
    x_input = generate_latent_points(latent_dim,n)
    #predict outputs
    X=generator.predict(x_input)
    #create class labels
    y = zeros((n,1))
    return X, y

#evaluate the discriminator an plot real and fake points
def summarize_performance(epoch,generator,discriminator,latent_dim,n=100):
    #prepare real samples
    x_real, y_real = generate_real_samples(n)
    #evaluate discriminator on real examples
    loss_real,acc_real = discriminator.evaluate(x_real,
                                                y_real,
                                                verbose = 0)
    #prepare fake examples
    x_fake, y_fake = generate_fake_samples(generator,
                                          latent_dim,
                                          n)
    #evaluate discriminator on fake examples
    loss_fake,acc_fake = discriminator.evaluate(x_real,
                                                y_real,
                                                verbose = 0)
    #summarize discriminator performace
    #print('epoch:{} with real accurancy: {} and fake accurancy: {}'.format(epoch,acc_real,acc_fake))
    #print('epoch:{} with real loss: {} and fake loss: {}'.format(epoch,loss_real,loss_fake))
    plt.scatter(x_real[:,0],x_real[:,1],color='pink')
    plt.scatter(x_fake[:,0],x_fake[:,1],color='blue')
    #save plot  and data to file
    datos.write('epoch:{} with real accurancy: {} and fake accurancy: {}\n'.format(epoch,acc_real,acc_fake))
    datos.write('epoch:{} with real loss: {} and fake loss: {}\n\n'.format(epoch,loss_real,loss_fake))
    filename = 'generated_plot_e%d.png' %(epoch+1)
    plt.savefig(filename)
    #plt.close()
    plt.show()
    
#train the GAN
def train(g_model,d_model,gan_model,latent_dim,
          n_epochs=10000,n_batch=128,n_eval=2000):
    #determine half the size of one batch for updating the discriminator
    half_batch = int(n_batch/2)

    #manually enumerate epochs
    for i in range(n_epochs):
        #prepare real examples
        x_real,y_real = generate_real_samples(half_batch)
        #prepare fake examples
        x_fake,y_fake =generate_fake_samples(g_model,
                                            latent_dim,
                                            half_batch)
        #update discriminator
        d_model.train_on_batch(x_real,y_real)
   
        d_model.train_on_batch(x_fake,y_fake)

        #prepare points in latent space as input for the generator
        x_gan =generate_latent_points(latent_dim,n_batch)
        #create inverted labels for the fake samples
        y_gan =ones((n_batch,1))
        #update the generator via the discriminator's error
        gan_model.train_on_batch(x_gan,y_gan)

        #evaluate the model every n_eval epochs
        if (i+1) % n_eval == 0:
            summarize_performance(i,g_model,d_model,latent_dim)

          

#size of the laten space
latent_dim =5
#create the discriminator
discriminator = define_discriminator()
#create the generator
generator = define_generator(latent_dim)
#create the gan
gan_model = define_gan(generator,discriminator)
#train model
train(generator,discriminator,gan_model,latent_dim,n_eval=1000)

datos.close()