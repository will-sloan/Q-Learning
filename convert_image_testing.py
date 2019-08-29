from PIL import Image
import cv2
import numpy as np

# img = Image.open('image.png')
# #img.show()
# print(img.size)
# img = img.convert("LA")
# #img.show()
# img = cv2.resize(numpy.array(img), (84,84,), interpolation=cv2.INTER_LINEAR)
# img2 = cv2.resize(numpy.array(img), (84,84))
# print(img.shape)
# print(img2.shape)
# img = Image.fromarray(img)
# img.show()
# print(img.size)

img = cv2.imread('image.png')
n = np.array(img)
n =np.expand_dims(n, axis=-1)
gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (84,84), interpolation=cv2.INTER_LINEAR)
grey = np.array(gray).reshape((1,84,84,1))
gray = np.expand_dims(gray, axis=-1)
print(grey.shape)
print(gray.shape)
print(n.shape)
#cv2.imshow("gray", gray)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
