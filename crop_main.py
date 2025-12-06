# Step 1: importing libraries
from tkinter import *
import tkinter
from tkinter import filedialog, messagebox

import os
import pickle
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageTk

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Dropout

# Step 2: variables used in full program

cnn_model = None
pca = None
crop_cls = None
crop_label_encoder = None
crop_csv_file = None

# Step 3: labels for diseases and fertilizers
leaf_labels = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
]

fertilizers = [
    "Mancozeb or Captan",
    "Copper-based fungicide",
    "Sulfur or Lime Sulfur",
    "No fertilizer needed",
    "Azoxystrobin",
    "Mancozeb for Common Rust",
    "No fertilizer needed",
    "Chlorothalonil",
    "Copper-based fungicide",
    "Thiophanate-methyl",
    "No fertilizer needed",
    "Mancozeb",
    "Copper oxychloride",
    "No fertilizer needed",
    "Metalaxyl or Mancozeb",
    "Copper-based fungicide",
    "Chlorothalonil",
    "No fertilizer needed",
    "Metalaxyl or Chlorothalonil",
    "Chlorothalonil",
    "Mancozeb for Septoria",
    "Insecticidal soap or Abamectin",
    "Difenoconazole",
    "Copper-based fungicide",
    "Imidacloprid or Acetamiprid",
]

# Step 4: color range for green mask
low_green = np.array([25, 52, 72])
high_green = np.array([102, 255, 255])

# Step 5: build main window
main = tkinter.Tk()
main.title("Optimizing Agriculture using Machine Learning")
main.geometry("1000x650")

# Step 6: set background
try:
    bg = Image.open("front.png")
    bg = bg.resize((1000, 650), Image.LANCZOS)
    bg_img = ImageTk.PhotoImage(bg)
    bg_label = tkinter.Label(main, image=bg_img)
    bg_label.place(relwidth=1, relheight=1)
except Exception:
    main.config(bg="light coral")

font_btn = ("times", 12, "bold")
font_txt = ("times", 12, "bold")

# Step 7: text box for messages
text = Text(main, height=15, width=80, font=font_txt)
text.place(x=250, y=200)

def log(msg: str):
    """Step 8: helper to print text."""
    text.insert(END, msg + "\n")
    text.see(END)
    text.update_idletasks()

# LEAF DISEASE FUNCTIONS

def load_leaf_dataset():
    """Step 9: load leaf features and apply PCA."""
    global X_leaf, Y_leaf, pca

    text.delete("1.0", END)

    # Step 9.1: opening folder so user knows which dataset
    _ = filedialog.askdirectory(
        parent=main,
        initialdir=".",
        title="Select PlantDiseaseDataset folder",
    )

    try:
        X = np.load(os.path.join("model", "X.txt.npy"))
        Y = np.load(os.path.join("model", "Y.txt.npy"))
    except Exception as e:
        log("Error loading model/X.txt.npy or Y.txt.npy")
        log(str(e))
        return

    # Step 9.2: scale and shuffle
    X = X.astype("float32") / 255.0
    Y = Y.astype("int32")
    sample_img = X[2].reshape(64, 64, 3)

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    X = X[idx]
    Y = Y[idx]

    # Step 9.3: PCA build or load
    pca_path = os.path.join("model", "pca.txt")
    if os.path.exists(pca_path):
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)
        X = pca.transform(X)
    else:
        pca = PCA(n_components=min(1200, X.shape[1]))
        X = pca.fit_transform(X)
        with open(pca_path, "wb") as f:
            pickle.dump(pca, f)

    Y_cat = to_categorical(Y)

    X_leaf = X
    Y_leaf = Y_cat

    log("Leaf dataset loaded.")
    log(f"Total images       : {X.shape[0]}")
    log(f"Features after PCA : {X.shape[1]}")

    cv2.imshow("Sample Leaf Image", cv2.resize(sample_img, (300, 300)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def train_leaf_cnn():
    """Step 10: train or load leaf CNN."""
    global cnn_model, X_leaf, Y_leaf

    if "X_leaf" not in globals() or "Y_leaf" not in globals():
        log("Upload leaf disease dataset first.")
        return

    text.delete("1.0", END)

    # Step 10.1: split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_leaf, Y_leaf, test_size=0.2, random_state=42
    )

    model_json_path = os.path.join("model", "model.json")
    weights_path = os.path.join("model", "model.weights.h5")

    cnn_model = None

    # Step 10.2: try load old model
    if os.path.exists(model_json_path) and os.path.exists(weights_path):
        try:
            with open(model_json_path, "r") as f:
                loaded_json = f.read()
            cnn_model = model_from_json(loaded_json)
            cnn_model.load_weights(weights_path)
            log("Loaded leaf CNN model from disk.")
        except Exception:
            cnn_model = None

    # Step 10.3: train new model if needed
    if cnn_model is None:
        log("Training new leaf CNN model...")

        cnn_model = Sequential()
        cnn_model.add(Dense(512, input_shape=(X_train.shape[1],), activation="relu"))
        cnn_model.add(Dropout(0.3))
        cnn_model.add(Dense(512, activation="relu"))
        cnn_model.add(Dropout(0.3))
        cnn_model.add(Dense(y_train.shape[1], activation="softmax"))

        cnn_model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        history = cnn_model.fit(
            X_train,
            y_train,
            batch_size=16,
            epochs=15,
            validation_data=(X_test, y_test),
            verbose=2,
        )

        with open(model_json_path, "w") as f:
            f.write(cnn_model.to_json())
        cnn_model.save_weights(weights_path)

        with open(os.path.join("model", "history.pckl"), "wb") as f:
            pickle.dump(history.history, f)

        log("Leaf CNN model trained and saved.")

    # Step 10.4: evaluate model
    y_pred = cnn_model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100
    fsc = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100

    log("Leaf CNN performance:")
    log(f"Accuracy : {acc:.2f}")
    log(f"Precision: {prec:.2f}")
    log(f"Recall   : {rec:.2f}")
    log(f"F1-score : {fsc:.2f}")

    # Step 10.5: confusion matrix
    conf = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        conf,
        xticklabels=leaf_labels,
        yticklabels=leaf_labels,
        annot=False,
        cmap="viridis",
        fmt="g",
    )
    ax.set_ylim([0, len(leaf_labels)])
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.title("CNN Leaf Disease Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.tight_layout()
    plt.show()


def classify_leaf_image():
    """Step 11: classify one leaf image."""
    global cnn_model, pca

    text.delete("1.0", END)

    if cnn_model is None or pca is None:
        log("Train leaf CNN before classification.")
        return

    # Step 11.1: choose image
    img_path = filedialog.askopenfilename(
        parent=main,
        initialdir="testImages",
        title="Select leaf image",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg")],
    )
    if not img_path:
        return

    img = cv2.imread(img_path)
    if img is None:
        log("Cannot read image.")
        return

    # Step 11.2: k-means segmentation
    Z = np.float32(img.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4
    _, labels_km, centers = cv2.kmeans(
        Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    labels_km = labels_km.reshape(img.shape[:-1])
    _ = np.uint8(centers)[labels_km]

    # Step 11.3: remove green background
    img_small = cv2.resize(img, (64, 64))
    imgHSV = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imgHSV, low_green, high_green)
    mask = 255 - mask
    res = cv2.bitwise_and(img_small, img_small, mask=mask)
    segmented = res

    # Step 11.4: apply PCA and predict
    res_flat = res.reshape(1, -1)
    try:
        feat = pca.transform(res_flat)
    except Exception:
        log("PCA mismatch, retrain model.")
        return

    feat = feat.astype("float32") / 255.0
    pred_probs = cnn_model.predict(feat)
    pred_idx = int(np.argmax(pred_probs))

    disease = leaf_labels[pred_idx]
    fert = fertilizers[pred_idx]

    log(f"Detected leaf disease : {disease}")
    log(f"Recommended fertilizer: {fert}")

    # Step 11.5: show result on image
    img_disp = cv2.resize(img, (800, 400))
    cv2.putText(
        img_disp,
        f"Disease   : {disease}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        img_disp,
        f"Fertilizer: {fert}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Leaf Disease Classification", img_disp)
    cv2.imshow("Segmented Image", cv2.resize(segmented, (200, 200)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# CROP RECOMMENDATION

def load_crop_dataset_and_train():
    """Step 12: load crop CSV and train decision tree."""
    global crop_cls, crop_label_encoder, crop_csv_file

    text.delete("1.0", END)

    # Step 12.1: choose CSV file
    csv_path = filedialog.askopenfilename(
        parent=main,
        initialdir=os.getcwd(),
        title="Select crop dataset CSV (cpdata.csv)",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    if not csv_path:
        log("Crop dataset not selected.")
        return

    crop_csv_file = csv_path
    log(f"Crop dataset file: {os.path.basename(csv_path)}")

    try:
        data = pd.read_csv(csv_path)
    except Exception as e:
        log("Error reading CSV file.")
        log(str(e))
        return

    try:
        # first 7 columns = features, 8th = label
        X = data.iloc[:, 0:7].values
        y = data.iloc[:, 7].values
    except Exception as e:
        log("CSV format error. Need 7 feature columns + 1 crop column.")
        log(str(e))
        return

    # Step 12.2: encode labels
    crop_label_encoder = LabelEncoder()
    y_encoded = crop_label_encoder.fit_transform(y)

    # Step 12.3: split and train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=0
    )

    crop_cls = DecisionTreeClassifier(random_state=0)
    crop_cls.fit(X_train, y_train)

    y_pred = crop_cls.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100

    log("Crop decision tree trained.")
    log(f"Dataset size : {len(data)}")
    log(f"Training size: {len(X_train)}")
    log(f"Test size    : {len(X_test)}")
    log(f"Accuracy     : {acc:.2f}%")


def predict_crop_popup():
    """Step 13: open popup and predict crop."""
    global crop_cls, crop_label_encoder

    if crop_cls is None or crop_label_encoder is None:
        log("Train crop model first (Crop Dataset button).")
        return

    popup = Toplevel(main)
    popup.title("Enter Test Data")
    popup.geometry("400x400")

    feature_names = ["Temperature", "Humidity", "pH", "Rainfall", "N", "P", "K"]
    entries = {}

    # Step 13.1: make input form
    for i, feat in enumerate(feature_names):
        Label(popup, text=feat, font=("Arial", 12)).grid(
            row=i, column=0, padx=10, pady=5, sticky="w"
        )
        ent = Entry(popup, font=("Arial", 12))
        ent.grid(row=i, column=1, padx=10, pady=5)
        entries[feat] = ent

    def on_submit():
        # Step 13.2: read user values
        try:
            vals = [float(entries[f].get()) for f in feature_names]
        except Exception:
            log("Enter numbers for all fields.")
            return

        test_arr = np.array(vals, dtype="float32").reshape(1, -1)

        # Step 13.3: run model and decode label
        y_pred_enc = crop_cls.predict(test_arr)[0]
        crop_name = crop_label_encoder.inverse_transform([y_pred_enc])[0]

        log(f"Predicted crop (Decision Tree): {crop_name}")

        msg = (
            "Crop Recommendation Result\n\n"
            "Input values:\n"
            f"  Temperature : {vals[0]}\n"
            f"  Humidity    : {vals[1]}\n"
            f"  pH          : {vals[2]}\n"
            f"  Rainfall    : {vals[3]}\n"
            f"  N           : {vals[4]}\n"
            f"  P           : {vals[5]}\n"
            f"  K           : {vals[6]}\n\n"
            f"Recommended crop: {crop_name}"
        )
        messagebox.showinfo("Crop Recommendation Result", msg)
        popup.destroy()

    Button(
        popup, text="Submit", command=on_submit, font=("Arial", 12), bg="lightblue"
    ).grid(row=len(feature_names), column=0, columnspan=2, pady=10)

    popup.mainloop()


# BUTTONS AND MAIN LOOP

Button(
    main,
    text="Upload Leaf Disease Dataset",
    command=load_leaf_dataset,
    font=font_btn,
).place(x=15, y=200)

Button(
    main,
    text="Train CNN Algorithm",
    command=train_leaf_cnn,
    font=font_btn,
).place(x=15, y=250)

Button(
    main,
    text="Disease Classification",
    command=classify_leaf_image,
    font=font_btn,
).place(x=15, y=300)

Button(
    main,
    text="Crop Dataset",
    command=load_crop_dataset_and_train,
    font=font_btn,
).place(x=15, y=350)

Button(
    main,
    text="Predict Crop",
    command=predict_crop_popup,
    font=font_btn,
).place(x=15, y=400)

# Step 14: start GUI
main.mainloop()
