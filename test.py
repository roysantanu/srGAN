###################################################################################
#Test - perform super resolution using saved generator model
from keras.models import load_model
from numpy.random import randint

generator = load_model('gen_e_10.h5', compile=False)


[X1, X2] = [lr_test, hr_test]
# select random example
ix = randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]

# generate image from source
gen_image = generator.predict(src_image)


# plot all three images

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('LR Image')
plt.imshow(src_image[0,:,:,:])
plt.subplot(232)
plt.title('Superresolution')
plt.imshow(gen_image[0,:,:,:])
plt.subplot(233)
plt.title('Orig. HR image')
plt.imshow(tar_image[0,:,:,:])

plt.show()


################################################
sr_lr = cv2.imread("data/sr_32.jpg")
sr_hr = cv2.imread("data/sr256.jpg")

#Change images from BGR to RGB for plotting. 
#Remember that we used cv2 to load images which loads as BGR.
sr_lr = cv2.cvtColor(sr_lr, cv2.COLOR_BGR2RGB)
sr_hr = cv2.cvtColor(sr_hr, cv2.COLOR_BGR2RGB)

sr_lr = sr_lr / 255.
sr_hr = sr_hr / 255.

sr_lr = np.expand_dims(sr_lr, axis=0)
sr_hr = np.expand_dims(sr_hr, axis=0)

generated_sr_hr = generator.predict(sr_lr)

# plot all three images
plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('LR Image')
plt.imshow(sr_lr[0,:,:,:])
plt.subplot(232)
plt.title('Superresolution')
plt.imshow(generated_sr_hr[0,:,:,:])
plt.subplot(233)
plt.title('Orig. HR image')
plt.imshow(sr_hr[0,:,:,:])

plt.show()