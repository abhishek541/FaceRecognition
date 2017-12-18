from PIL import Image
from os import listdir
import numpy as np

"""

A Python class that implements the Eigenfaces algorithm
for face recognition, using eigenvalues and
principle component analysis. A model is trained
with some images and then it is used to predict if
given image is a face image and if it matches with any
of the training images

"""

class FaceRecognition:
    rows = 195
    cols = 231
    T0 = 6500000000000
    T1 = 89000000

    """
    Initializing the class
    """
    def __init__(self):
        pass

    """
    Calculate difference between given image and mean face
    """
    def diff_mean_face(self, img):
        img -= self.mean_face[:]
        return img

    """
    Returns the projection of image on the face space
    """
    def compute_projection(self, img):
        proj_face = np.dot(self.eigen_faces.transpose(), img)
        return proj_face

    """
    Returns reconstructed image from eigen faces
    """
    def reconstruct_image(self, omega):
        img_proj = np.dot(self.eigen_faces, omega)
        return img_proj

    """
    Get face space from training images
    """
    def get_face_space(self):
        A_transpose = np.transpose(self.A)
        L = np.dot(A_transpose, self.A)
        eig_val, eig_vec = np.linalg.eig(L)
        U = np.dot(self.A, eig_vec)
        return U

    """
    Calculate Euclidean distance between 2 arrays
    """
    def get_euclidean_distance(self, arr1, arr2):
        dist_arr = np.linalg.norm(np.subtract(arr1, arr2))
        return dist_arr

    """
    Save data as image to the specified path
    """
    def save_image(self, imgdata, filepath, size):
        im = Image.new('L', size)
        im.putdata(imgdata)
        im.save(filepath)

    """
    Get all the images from train directory
    Calculate the mean face and subtract from each image
    Get eigen faces and projection on face space
    """
    def train_eigen_faces(self):
        train_files = listdir('train')
        self.A = np.zeros(shape=(self.rows * self.cols, len(train_files)), dtype='int64')
        print(str(train_files))
        for i in range(0, len(train_files)):
            img = Image.open('train/' + train_files[i])
            img_arr = np.array(img, dtype='int64').flatten()
            self.A[:, i] = img_arr[:]

        self.mean_face = np.floor_divide(np.sum(self.A, axis=1) ,len(train_files))
        self.save_image(self.mean_face, 'output/mean_face.png', (self.rows, self.cols))
        for j in range(0, len(train_files)):
            self.A[:, j] -= self.mean_face[:]
            self.save_image(self.A[:, j], 'output/train_diff'+str(j)+'.png', (self.rows, self.cols))

        self.eigen_faces = self.get_face_space()
        for k in range(0, self.eigen_faces.shape[1]):
            self.save_image(self.eigen_faces[:, k], 'output/eigen_face'+str(k)+'.png', (self.rows, self.cols))

        train_projections = []
        for l in range(0, len(train_files)):
            projection = np.dot(self.eigen_faces.transpose(), self.A[:, l])
            print('Projection for training image ' + str(l) + ' is: ' + str(projection))
            train_projections.append(projection)
        self.train_proj = np.array(train_projections)
        print('---------------End of training------------------')

    """
    For each image, predict if it is a face image
    Predict if the face matches with any of the training images
    """
    def predict_image(self, test_img, n):
        diff_test = self.diff_mean_face(test_img)
        self.save_image(diff_test, 'output/test_diff'+str(n)+'.png', (self.rows, self.cols))
        test_proj = self.compute_projection(diff_test)
        print('Image projection: ' + str(test_proj))
        test_recons = self.reconstruct_image(test_proj)
        self.save_image(test_recons, 'output/test_recons'+str(n)+'.png', (self.rows, self.cols))
        test_distance = self.get_euclidean_distance(test_recons, diff_test)
        dist_test_train = []
        for omega in self.train_proj:
            euc_dist = self.get_euclidean_distance(test_proj, omega)
            dist_test_train.append(euc_dist)

        print("Test distance is: " + str(test_distance))
        print("Test train distances are: " + str(dist_test_train))
        print("Minimum test train distance is: " + str(min(dist_test_train)))
        if test_distance > self.T0:
            print("Not a face image")
        else:
            if min(dist_test_train) < self.T1:
                print("Face recognised with training image " + str(dist_test_train.index(min(dist_test_train))+1))
            else:
                print("Face not recognised")
        print("===================================================")

"""
Method to call the FaceRecognition class for training and prediction
"""
def test():
    fr = FaceRecognition()
    fr.train_eigen_faces()
    test_files = listdir('test')
    for i in range(len(test_files)):
        print("File is: " + test_files[i])
        img = Image.open('test/' + test_files[i])
        img_arr = np.array(img, dtype='int64').flatten()
        fr.predict_image(img_arr, i)

if __name__ == "__main__":
    test()