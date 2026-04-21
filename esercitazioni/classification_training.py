import skimage.data as data
import skimage.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import numpy as np
from sklearn.svm import LinearSVC
from skimage.feature import local_binary_pattern
from glob import glob

def main():
    train_label=np.load('images/breakhis/train_label.npy')
    train_feat=np.load('images/breakhis/train_lbp_8_1_default.npy')
    
    #Normalizzazione (Z-score)
    mu=np.mean(train_feat,0)
    sigma=np.std(train_feat,0)
    train_feat=(train_feat-mu)/(sigma+1e-15)
    
    #Addestramento
    classifier=LinearSVC().fit(train_feat, train_label)

    #Preparazione Test Set
    list_files_benigni=glob("images/breakhis/testset/benign/*.png")
    list_files_maligni=glob("images/breakhis/testset/malignant/*.png")
    
    #Benigno = 0, Maligno = 1
    test_files=list_files_benigni+list_files_maligni
    test_labels_gt=np.array([0] * len(list_files_benigni) + [1] * len(list_files_maligni))
    
    test_preds=[]

    print(f"Inizio test su {len(test_files)} immagini...")

    #Ciclo di Testing
    for file_path in test_files:
        #Lettura e calcolo LBP
        img=io.imread(file_path)
            
        lbp=local_binary_pattern(img, 8, 1, method='default')
        
        #Istogramma (Feature extraction)
        feat,_=np.histogram(lbp.flatten(), bins=np.arange(0, 257), density=True)
        
        #Normalizzazione
        feat=(feat-mu)/(sigma+1e-15)
        feat=np.reshape(feat,(1, -1))
        
        #Predizione
        pred=classifier.predict(feat)
        test_preds.append(pred[0])

    test_preds=np.array(test_preds)
    accuracy=np.mean(test_preds == test_labels_gt) * 100
    
    print("--- RISULTATI TEST ---")
    print(f"Accuracy totale: {accuracy:.2f}%")
    
    from sklearn.metrics import classification_report
    print("\nReport di classificazione:")
    print(classification_report(test_labels_gt, test_preds, target_names=['Benigno', 'Maligno']))

if __name__=='__main__':
    main()