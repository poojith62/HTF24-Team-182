from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev-key-123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///wardrobe.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Initialize ML model as None
model = None

def load_ml_model():
    global model
    if model is None:
        model = MobileNetV2(weights='imagenet', include_top=True)

# Create a function to initialize the app
def create_app():
    # Ensure upload folder exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # Create database tables
    with app.app_context():
        db.create_all()
        load_ml_model()
    
    return app

# Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    items = db.relationship('ClothingItem', backref='owner', lazy=True)

class ClothingItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    color = db.Column(db.String(50), nullable=False)
    image_path = db.Column(db.String(200))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# ML-related configurations remain the same
CLOTHING_CATEGORY_MAP = {
    'jersey': 'top',
    'sweater': 'top',
    'sweatshirt': 'top',
    'cardigan': 'top',
    'shirt': 'top',
    'tee_shirt': 'top',
    'jean': 'bottom',
    'pants': 'bottom',
    'trouser': 'bottom',
    'skirt': 'bottom',
    'dress': 'dress',
    'gown': 'dress',
    'coat': 'outerwear',
    'jacket': 'outerwear',
    'running_shoe': 'shoes',
    'shoe': 'shoes',
    'sneaker': 'shoes',
    'boot': 'shoes',
    'sandal': 'shoes',
}

COLOR_MAP = {
    'black': ([0, 0, 0], 30),
    'white': ([255, 255, 255], 30),
    'red': ([255, 0, 0], 50),
    'blue': ([0, 0, 255], 50),
    'green': ([0, 255, 0], 50),
    'yellow': ([255, 255, 0], 50),
    'purple': ([128, 0, 128], 50),
    'orange': ([255, 165, 0], 50),
    'brown': ([165, 42, 42], 50),
    'gray': ([128, 128, 128], 50),
}

# Helper function for color complementary checking
# Helper function for color complementary checking
def is_complementary(color1, color2):
    complementary_pairs = {
        'black': ['white', 'gray'],  # Neutral, goes with most colors
        'white': ['black', 'gray', 'blue', 'red'],  # Often paired with black or vibrant colors
        'red': ['green', 'blue', 'gray'],  # Complements green and blue
        'blue': ['orange', 'yellow', 'brown'],  # Complements orange, yellow, brown
        'green': ['red', 'purple', 'brown'],  # Complements red and brown
        'yellow': ['purple', 'blue', 'gray'],  # Complements purple, blue
        'purple': ['yellow', 'green', 'orange'],  # Complements yellow and green
        'orange': ['blue', 'purple', 'white'],  # Complements blue, purple, white
        'brown': ['blue', 'green', 'gray'],  # Complements blue, green, and gray
        'gray': ['black', 'white', 'red', 'yellow', 'blue'],  # Complements most colors, versatile
    }
    return color2 in complementary_pairs.get(color1, [])


def extract_dominant_color(image_path):
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize((100, 100))
        img_array = np.array(img)
        img_array = img_array.reshape(-1, 3)

        kmeans = KMeans(n_clusters=3)
        kmeans.fit(img_array)
        dominant_color = kmeans.cluster_centers_[0]
        
        min_distance = float('inf')
        closest_color = 'multicolor'
        
        for color_name, (color_value, threshold) in COLOR_MAP.items():
            distance = np.sqrt(np.sum((dominant_color - np.array(color_value)) ** 2))
            if distance < min_distance and distance < threshold:
                min_distance = distance
                closest_color = color_name
                
        return closest_color
    except Exception as e:
        print(f"Error in color extraction: {e}")
        return 'multicolor'

def classify_clothing(image_path):
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        preds = model.predict(img_array)
        decoded_preds = decode_predictions(preds, top=5)[0]
        
        for pred in decoded_preds:
            predicted_class = pred[1].lower()
            if predicted_class in CLOTHING_CATEGORY_MAP:
                return CLOTHING_CATEGORY_MAP[predicted_class]
        
        return 'other'
    except Exception as e:
        print(f"Error in clothing classification: {e}")
        return 'other'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
            
        user = User(
            username=username,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()
        flash('Registration successful!')
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('wardrobe'))
            
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/wardrobe')
@login_required
def wardrobe():
    # Get filter parameters
    category_filter = request.args.get('category', 'all')
    color_filter = request.args.get('color', 'all')
    
    # Base query
    query = ClothingItem.query.filter_by(user_id=current_user.id)
    
    # Apply filters if they're set
    if category_filter != 'all':
        query = query.filter_by(category=category_filter)
    if color_filter != 'all':
        query = query.filter_by(color=color_filter)
    
    # Get all items
    items = query.all()
    
    # Organize items by category
    categorized_items = {
        'top': [],
        'bottom': [],
        'dress': [],
        'outerwear': [],
        'shoes': [],
        'other': []
    }
    
    # Get unique colors for filter
    all_colors = set()
    
    for item in items:
        categorized_items[item.category].append(item)
        all_colors.add(item.color)
    
    return render_template('wardrobe.html',
                         categorized_items=categorized_items,
                         all_colors=sorted(list(all_colors)),
                         current_category=category_filter,
                         current_color=color_filter)
@app.route('/add_item', methods=['GET', 'POST'])
@login_required
def add_item():
    if request.method == 'POST':
        filename = request.form.get('filename')
        name = request.form.get('name')
        category = request.form.get('category')
        color = request.form.get('color')
        
        if not all([filename, name, category, color]):
            flash('Please fill in all fields')
            return redirect(url_for('add_item'))
            
        try:
            # Create new clothing item - store relative path
            new_item = ClothingItem(
                name=name,
                category=category,
                color=color,
                # Store path relative to static folder
                image_path=filename,  # Changed from os.path.join('uploads', filename)
                user_id=current_user.id
            )
            
            db.session.add(new_item)
            db.session.commit()
            
            flash(f"Item '{name}' has been added successfully!")
            return redirect(url_for('wardrobe'))
            
        except Exception as e:
            print(f"Error adding item to database: {str(e)}")
            flash('Error saving item. Please try again.')
            return redirect(url_for('add_item'))

    return render_template('add_item.html')

@app.route('/detect_item', methods=['POST'])
def detect_item():
    print("=== Starting Item Detection ===")
    
    # Check if image was uploaded
    if 'image' not in request.files:
        print("Error: No image file in request")
        flash('No image uploaded')
        return redirect(url_for('add_item'))

    file = request.files['image']
    if file.filename == '':
        print("Error: Empty filename")
        flash('No file selected')
        return redirect(url_for('add_item'))

    if not allowed_file(file.filename):
        print(f"Error: Invalid file type for {file.filename}")
        flash('Invalid file type. Please upload a PNG, JPG, or JPEG image.')
        return redirect(url_for('add_item'))

    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(f"File saved successfully at: {file_path}")

        # Ensure ML model is loaded
        load_ml_model()
        print("ML model loaded successfully")

        # Perform clothing classification
        print("Starting clothing classification...")
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        preds = model.predict(img_array)
        decoded_preds = decode_predictions(preds, top=5)[0]
        
        print("\nTop 5 predictions:")
        detected_name = None
        detected_category = 'other'
        
        for i, (imagenet_id, class_name, score) in enumerate(decoded_preds):
            print(f"{i+1}. {class_name}: {score*100:.2f}%")
            if class_name.lower() in CLOTHING_CATEGORY_MAP and not detected_name:
                detected_name = class_name.lower()
                detected_category = CLOTHING_CATEGORY_MAP[class_name.lower()]
                print(f"Found matching clothing item: {detected_name} in category: {detected_category}")

        # Extract dominant color
        print("\nExtracting dominant color...")
        detected_color = extract_dominant_color(file_path)
        print(f"Detected color: {detected_color}")

        # If no clothing item was detected in top 5 predictions
        if not detected_name:
            print("Warning: No specific clothing item detected in top predictions")
            detected_name = "item"

        return render_template('add_item.html',
                             filename=filename,
                             detected_category=detected_category,
                             detected_color=detected_color,
                             detected_name=detected_name)

    except Exception as e:
        print(f"Error during detection: {str(e)}")
        import traceback
        traceback.print_exc()
        flash('Error during item detection. Please try again.')
        return redirect(url_for('add_item'))
    




    
from flask import render_template, flash
from flask_login import login_required, current_user
import random



@app.route('/suggestions')
@login_required
def suggestions():
    # Retrieve user's clothing items from the database based on category
    tops = ClothingItem.query.filter_by(user_id=current_user.id, category='top').all()
    bottoms = ClothingItem.query.filter_by(user_id=current_user.id, category='bottom').all()
    shoes = ClothingItem.query.filter_by(user_id=current_user.id, category='shoes').all()

    # Check if there are enough items to create at least one complete outfit
    if not tops or not bottoms or not shoes:
        flash("Please add at least one top, one bottom, and one pair of shoes to get outfit suggestions.")
        return render_template('suggestions.html', outfits=[], color_theory_outfits=[])

    # List to hold classic combinations (one top, one bottom, one shoe per outfit)
    outfits = []

    # Generate up to 5 unique outfits for the Classic Combinations section
    max_combinations = min(5, len(tops) * len(bottoms) * len(shoes))
    for _ in range(max_combinations):
        # Randomly select one top, one bottom, and one shoe for each outfit
        top = random.choice(tops)
        bottom = random.choice(bottoms)
        shoe = random.choice(shoes)
        
        # Append the outfit as a dictionary for easier access in the template
        outfits.append({
            'top': top,
            'bottom': bottom,
            'shoe': shoe
        })

    # Create color theory outfits for the "Color Combinations" section (optional section based on user needs)
    color_theory_outfits = []
    for _ in range(max_combinations):
        top = random.choice(tops)
        bottom = random.choice(bottoms)
        selected_shoes = random.sample(shoes, min(3, len(shoes)))  # Select up to 3 shoes for variety

        # Append outfit with multiple shoes for the Color Theory section
        color_theory_outfits.append({
            'top': top,
            'bottom': bottom,
            'shoes': selected_shoes  # List of shoes for the carousel display
        })

    # Render the suggestions page with both classic and color theory outfits
    return render_template(
        'suggestions.html',
        outfits=outfits,                # For Classic Combinations section
        color_theory_outfits=color_theory_outfits  # For Color Theory Combinations section
    )


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)