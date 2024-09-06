import os
import shutil
from collections import Counter
from PIL import Image
import logging
import gradio as gr
import torch
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from tqdm import tqdm  # For progress bar
import clip
from multiprocessing import Pool, cpu_count
from model_loader import ModelLoader
from keyword_extractor import KeywordExtractor
from classifier import Classifier
from config import Config


# --- Configuration ---
class Config:
    def __init__(self):
        self.input_dir = ""
        self.output_dir = ""
        self.keyword_extraction_method = "clip"  # Default to CLIP
        self.confidence_threshold = 0.8
        self.output_dirs = {
            "nature": "Nature",
            "people": "People",
            "animals": "Animals",
            "objects": "Objects",
            "other": "Other"
        }
        self.potential_keywords = [
            "landscape", "mountain", "tree", "forest", "river", "ocean", "sky", "water",
            "person", "man", "woman", "child", "face",
            "dog", "cat", "bird", "animal", "pet",
            "car", "building", "house", "object"
        ]
        
        # Pre-compute embeddings for potential keywords (for efficiency)
        self.potential_keywords_embeddings = None
        self.keyword_extractor = SentenceTransformer('all-mpnet-base-v2').to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# --- Model Loading ---
class ModelLoader:
    def __init__(self, device):
        self.device = device
        self.image_model = models.resnet101(pretrained=True).to(device)
        self.image_model.eval()
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)

# --- Keyword Extraction ---
class KeywordExtractor:
    def __init__(self, config, model_loader):
        self.config = config
        self.model_loader = model_loader
        self.config.potential_keywords_embeddings = self.config.keyword_extractor.encode(
            self.config.potential_keywords, convert_to_tensor=True).cpu().numpy()

    def extract_keywords(self, image_path, top_k=5):
        if self.config.keyword_extraction_method == "clip":
            return self.extract_keywords_clip(image_path, top_k)
        elif self.config.keyword_extraction_method == "baseline":
            return self.extract_keywords_baseline(image_path, top_k)
        else:
            logging.warning(f"Invalid keyword extraction method: {self.config.keyword_extraction_method}. Using baseline.")
            return self.extract_keywords_baseline(image_path, top_k)

    def extract_keywords_baseline(self, image_path, top_k=5):
        try:
            img = Image.open(image_path)
            img_tensor = preprocess(img).unsqueeze(0).to(self.model_loader.device)

            with torch.no_grad():
                features = self.model_loader.image_model(img_tensor)

            embeddings = features.cpu().numpy()
            keywords_embeddings = self.config.keyword_extractor.encode(embeddings, convert_to_tensor=True).cpu().numpy()

            # Use cosine similarity for better keyword matching
            similarities = cosine_similarity(keywords_embeddings, self.config.potential_keywords_embeddings)
            top_indices = np.argsort(similarities[0])[::-1][:top_k]
            keywords = [self.config.potential_keywords[i] for i in top_indices]

            return keywords
        except Exception as e:
            logging.error(f"Error extracting keywords from {image_path}: {e}")
            return []

    def extract_keywords_clip(self, image_path, top_k=5):
        try:
            image = self.model_loader.clip_preprocess(Image.open(image_path)).unsqueeze(0).to(self.model_loader.device)
            with torch.no_grad():
                image_features = self.model_loader.clip_model.encode_image(image)

            # Generate a set of potential keywords
            potential_keywords = self.config.potential_keywords

            with torch.no_grad():
                text_features = self.model_loader.clip_model.encode_text(clip.tokenize(potential_keywords)).float()

            # Calculate cosine similarity between image features and text features
            similarities = image_features @ text_features.T
            top_indices = torch.topk(similarities, top_k).indices.cpu().numpy()
            keywords = [potential_keywords[i] for i in top_indices]

            return keywords
        except Exception as e:
            logging.error(f"Error extracting keywords from {image_path} using CLIP: {e}")
            return []


# --- Classifier ---
class Classifier:
    def __init__(self, config, model_loader):
        self.config = config
        self.model_loader = model_loader
        self.classifier = None  # Initialize the trained classifier here


    def train_classifier(self, image_dir):
        images = []
        labels = []

        for filename in os.listdir(image_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(image_dir, filename)
                # Extract keywords
                keywords = KeywordExtractor(self.config, self.model_loader).extract_keywords(filepath, top_k=5)

                # Get the feature vector of the image
                img = Image.open(filepath)
                img_tensor = preprocess(img).unsqueeze(0).to(self.model_loader.device)
                with torch.no_grad():
                    features = self.model_loader.image_model(img_tensor)
                features = features.cpu().numpy().flatten()
                images.append(features)

                # Assign category based on folder name of the image
                folder = os.path.basename(os.path.dirname(filepath))
                if folder in self.config.output_dirs.keys():
                    labels.append(list(self.config.output_dirs.keys()).index(folder))
                else:
                    labels.append(len(self.config.output_dirs))

        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

        # Define parameter grid for hyperparameter tuning
        param_grid = {
            'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']  # Choose solvers suitable for L1 and L2
        }

        # Perform Grid Search with Cross-Validation
        grid_search = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5)
        grid_search.fit(X_train, y_train)

        # Get the best model and print the best parameters
        self.classifier = grid_search.best_estimator_  # Store the best model
        print(f"Best parameters: {grid_search.best_params_}")

        # Evaluate the classifier on the test set
        y_pred = self.classifier.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=list(self.config.output_dirs.keys())))

    def classify_image(self, image_path):
        try:
            img = Image.open(image_path)
            img_tensor = preprocess(img).unsqueeze(0).to(self.model_loader.device)
            with torch.no_grad():
                features = self.model_loader.image_model(img_tensor)
            features = features.cpu().numpy().flatten()
            predicted_label_probs = self.classifier.predict_proba([features])[0]
            predicted_label = np.argmax(predicted_label_probs)
            confidence = np.max(predicted_label_probs)

            if confidence >= self.config.confidence_threshold:
                return predicted_label
            else:
                return len(self.config.output_dirs)  # Default to "other" if confidence is too low
        except Exception as e:
            logging.error(f"Error classifying image {image_path}: {e}")
            return len(self.config.output_dirs)  # Default to "other" in case of error


# --- Image Processing ---
class ImageProcessor:
    def __init__(self, config, model_loader, keyword_extractor, classifier):
        self.config = config
        self.model_loader = model_loader
        self.keyword_extractor = keyword_extractor
        self.classifier = classifier

    def process_images(self, input_dir, output_dir):
            for root, _, files in os.walk(input_dir):  # Use os.walk for recursive traversal
                for filename in tqdm(files, desc="Processing Images"):
                    if filename.endswith(('.jpg', '.jpeg', '.png')):
                        filepath = os.path.join(root, filename)

                        # Classify image with the trained classifier
                        predicted_label = self.classifier.classify_image(filepath)

                        # Determine output directory based on predicted label
                        category = list(self.config.output_dirs.keys())[predicted_label]
                        final_output_dir = os.path.join(output_dir, self.config.output_dirs[category])
                        os.makedirs(final_output_dir, exist_ok=True)

                        # Rename file based on extracted keywords
                        keywords = self.keyword_extractor.extract_keywords(filepath, top_k=5)
                        new_filename = "_".join(keywords[:3]) + os.path.splitext(filename)[1]

                        try:
                            shutil.move(filepath, os.path.join(final_output_dir, new_filename))
                            logging.info(f"Processed {filename}:")
                            logging.info(f" - Predicted category: {category}")
                            logging.info(f" - Moved to: {final_output_dir}/{new_filename}")
                        except Exception as e:
                            logging.error(f"Error moving file {filename}: {e}")

    def process_images_multiprocessing(self, input_dir, output_dir):
        image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        with Pool(processes=cpu_count()) as pool:
            pool.starmap(self.process_image, [(input_dir, output_dir, f) for f in image_files])

    def process_image(self, input_dir, output_dir, filename):
        filepath = os.path.join(input_dir, filename)

        # Classify image with the trained classifier
        predicted_label = self.classifier.classify_image(filepath)

        # Determine output directory based on predicted label
        category = list(self.config.output_dirs.keys())[predicted_label]
        final_output_dir = os.path.join(output_dir, self.config.output_dirs[category])
        os.makedirs(final_output_dir, exist_ok=True)

        # Rename file based on extracted keywords
        keywords = self.keyword_extractor.extract_keywords(filepath, top_k=5)
        new_filename = "_".join(keywords[:3]) + os.path.splitext(filename)[1]

        try:
            shutil.move(filepath, os.path.join(final_output_dir, new_filename))
            logging.info(f"Processed {filename}:")
            logging.info(f" - Predicted category: {category}")
            logging.info(f" - Moved to: {final_output_dir}/{new_filename}")
        except Exception as e:
            logging.error(f"Error moving file {filename}: {e}")


# Image transformation (Add this to keyword_extractor.py)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Gradio Interface ---
def main():
    config = Config()
    model_loader = ModelLoader(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    keyword_extractor = KeywordExtractor(config, model_loader)
    classifier = Classifier(config, model_loader)  # Create Classifier instance
    image_processor = ImageProcessor(config, model_loader, keyword_extractor, classifier)  # Create ImageProcessor instance


    with gr.Blocks() as demo:
        gr.Markdown("## Image Sorter")

        with gr.Row():
            input_dir = gr.Textbox(label="Input Directory", placeholder="Path to images", interactive=True)
            input_dir_button = gr.Button("Browse", interactive=True)

        with gr.Row():
            output_dir = gr.Textbox(label="Output Directory", placeholder="Path to save sorted images", interactive=True)
            output_dir_button = gr.Button("Browse", interactive=True)

        with gr.Row():
            train_button = gr.Button("Train Classifier", interactive=True)

        with gr.Row():
            method_label = gr.Label("Keyword Extraction Method:")
            method = gr.Radio(choices=["CLIP", "Baseline"], value="CLIP", label="Method")

        with gr.Row():
            sort_button = gr.Button("Start Sorting", interactive=True)

        status = gr.Textbox(label="Status", interactive=False)

        # Event Handlers
        input_dir_button.click(lambda: input_dir.value, inputs=input_dir_button, outputs=input_dir)
        output_dir_button.click(lambda: output_dir.value, inputs=output_dir_button, outputs=output_dir)

        train_button.click(
            lambda input_dir: classifier.train_classifier(input_dir),  # Call train_classifier from classifier instance
            inputs=input_dir,
            outputs=classifier,
        )

        sort_button.click(
            lambda input_dir, output_dir: image_processor.process_images(input_dir, output_dir),  # Call process_images through ImageProcessor
            inputs=[input_dir, output_dir],
            outputs=status,
        )

    demo.launch()

# Remove these functions as they are now implemented within classes
# def train_classifier(...):
#     ...
# def sort_images(...):
#     ...
# def process_images(...):
#     ...

if __name__ == "__main__":
    main()