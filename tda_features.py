from skimage.transform import resize
from sklearn.decomposition import KernelPCA
from vizualisation import plot_persistence_images
from data_utils import get_features_and_labels, get_train_test_val
import numpy as np 
from persim import PersistenceImager
from ripser import ripser
pimager = PersistenceImager(pixel_size=0.2)

def get_tda_features_and_labels(dataframe_):
    ids = dataframe_['Unnamed']
    dataframe_['second_ids'] = ids.str.extract(r'(\d+)')[0].astype(int)
    dataframe_['subject_ids'] = ids.str.extract(r'(\d+$)')[0].astype(int)

    y, X = get_features_and_labels(dataframe_)

    subject_ids = dataframe_['subject_ids'].values

    X_train, y_train, X_val, y_val, X_test, y_test, sid_train, sid_val, sid_test = get_train_test_val(X, y, subject_ids)
    
  

   # 0.76
    from sklearn.decomposition import MiniBatchDictionaryLearning
    mini_dict = MiniBatchDictionaryLearning(n_components=3, batch_size=200, n_iter=300)
    X_train = mini_dict.fit_transform(X_train)
    X_val = mini_dict.transform(X_val)
    X_test = mini_dict.transform(X_test)
    
    labels_train, pim_train = compute_tda(X_train, y_train, sid_train, plot=True)
    labels_val, pim_val = compute_tda(X_val, y_val, sid_val)
    labels_test, pim_test = compute_tda(X_test, y_test, sid_test)
    
    return pim_train, labels_train, pim_val, labels_val, pim_test, labels_test


def diagram_to_persistence_image(pd, pim, birth_range, pers_range, subj, blank_images_counter):
    if len(pd) == 0:
        blank_images_counter[0] += 1
        return np.zeros((10, 10))
    pd_persistence = np.stack([pd[:, 0], pd[:, 1] - pd[:, 0]], axis=1)
    pd_clipped = np.clip(
        pd_persistence,
        a_min=(birth_range[0], 0),
        a_max=(birth_range[1], pers_range[1])
    )
    try:
        pimg = pim.transform(pd_clipped)
        if pimg is None or not np.isfinite(pimg).all() or pimg.size == 0:
            raise ValueError("Invalid persistence image")
        pimg = resize(pimg, (10, 10))
    except Exception as e:
        print(f"[Subject {subj}] Error during persistence image transformation: {e}")
        blank_images_counter[0] += 1
        pimg = np.zeros((10, 10))
    return pimg

def compute_tda(X, y, subject_ids=None, pixel_size=0.2, plot=False):
    birth_range = (0, 5)
    pers_range = (0, 5)

    pim = PersistenceImager(
        birth_range=birth_range,
        pers_range=pers_range,
        pixel_size=pixel_size,
        kernel_params={'sigma': 0.5}
    )

    persistence_images = []
    labels = []
    blank_images_counter = [0] 

    if subject_ids is None:
        subject_ids = np.array(['all'] * len(X))
    else:
        subject_ids = np.array(subject_ids)

    unique_subjects = np.unique(subject_ids)

    for subj in unique_subjects:
        mask = subject_ids == subj
        X_subj = X[mask]
        y_subj = y[mask]

        ph_result = ripser(X_subj, maxdim=1)
        pd_h0 = ph_result['dgms'][0]
        pd_h1 = ph_result['dgms'][1] if len(ph_result['dgms']) > 1 else np.zeros((0, 2))

        if len(pd_h0) > 0 and len(pd_h1) > 0:
            h0_range = pd_h0[:, 1] - pd_h0[:, 0]
            h0_mean = np.mean(h0_range) if len(h0_range) > 0 else 1
            h1_range = pd_h1[:, 1] - pd_h1[:, 0]
            scaled_pers = h0_mean * (h1_range / (np.max(h1_range) if np.max(h1_range) > 0 else 1))
            pd_h1[:, 1] = pd_h1[:, 0] + scaled_pers

        pimg_h0 = diagram_to_persistence_image(pd_h0, pim, birth_range, pers_range, subj, blank_images_counter)
        pimg_h1 = diagram_to_persistence_image(pd_h1, pim, birth_range, pers_range, subj, blank_images_counter)

        combined_pimg = np.concatenate([pimg_h0.flatten(), pimg_h1.flatten()])
        persistence_images.append(combined_pimg)

        if hasattr(y_subj, 'iloc'):
            labels.append(y_subj.iloc[0])
        else:
            labels.append(y_subj[0])

    if plot:
        plot_persistence_images(labels, persistence_images)

    return np.array(labels), np.array(persistence_images)