## Overview
A hybrid recommendation system combining Flask backend with React frontend. Features include various filtering techniques, a chatbot interface, and a social media-like user experience.

## Features
- **Backend**: Flask APIs implementing hybrid recommendation algorithms
- **Frontend**: React-based dynamic user interface
- **Recommendation Algorithms**: 
  - Collaborative filtering
  - Content-based filtering
  - NCF Based Hybrid system
- **Interactive UI**: User-friendly interface with comments, profiles, and home feed
- **Data Processing**: Pandas and NumPy integration
- **Machine Learning**: PyTorch and Transformers for recommendation models

## Prerequisites

### Backend Requirements
- Python 3.7 or higher
- Required Python packages:
  ```bash
  pip install flask pandas python-dotenv transformers numpy torch datasets scikit-learn scipy
  ```

### Frontend Requirements
- Node.js (16.x or later recommended)
- NPM (comes with Node.js)

## Installation

Make A File Under This Folder, Like This :
WEBDEV/server/combined_embeddings.npy

Download File From Here :
https://drive.google.com/file/d/1tsCP3zYCchn2MkGmqhdbIKWG4x9PU28Y/view

### Backend Setup

1. Create and activate virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install flask pandas python-dotenv transformers numpy torch datasets scikit-learn scipy
```

3. Start Flask server:
```bash
python server.py
```

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd WEBDEV
```

2. Install dependencies:
```bash
npm install
```

3. Start React development server:
```bash
npm start
```

## Project Structure
```
.
├── NCF_hybrid_recommendation.py
├── chatbotrecommender.ipynb
├── collaborative_filtering.py
├── content_based_filtering.py
├── embedding_based_hybrid_filtering.py
├── file_loader.py
├── WEBDEV
│   ├── package.json
│   ├── public
│   │   ├── index.html
│   │   └── manifest.json
│   ├── server
│   │   ├── server.py
│   │   ├── user_likes.xlsx
│   │   └── image_likes.xlsx
│   └── src
│       ├── App.js
│       ├── Components
│       │   ├── Home
│       │   │   ├── Homepage.js
│       │   │   └── Feedposts.js
│       ├── Pages
│       │   └── Home
│       │       ├── Home.js
│       │       └── Chatbot.jsx
│       └── index.js
```

## Technologies Used

### Backend
- Flask
- Python libraries:
  - pandas
  - python-dotenv
  - transformers
  - numpy
  - PyTorch
  - datasets
  - scikit-learn
  - scipy

### Frontend
- React.js
- Node.js
- NPM packages (defined in package.json)

## Getting Started

1. Clone the repository
2. Follow the Backend Setup instructions
3. Follow the Frontend Setup instructions
4. Access the application at `http://localhost:3000`

## Notes
- Ensure all required Python packages are installed before running the backend
- The frontend development server runs on port 3000 by default
- The Flask backend server should be running simultaneously with the frontend

# DEMO 
https://drive.google.com/file/d/1d9XqGEEgRTkqlYxNz-tHt-RkJrIK2Qbz/view?usp=sharing
