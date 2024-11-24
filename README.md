# ArtWork-DL-PROJECT

# Project Title

## Overview

This project combines a Flask backend with a React frontend to implement a hybrid recommendation system. It includes various filtering techniques, a chatbot interface, and a social media-like user experience.

---

## Features

- **Backend**: Implements Flask APIs for hybrid recommendation algorithms.
- **Frontend**: Developed using React for a dynamic user interface.
- **Recommendation Algorithms**: Includes collaborative filtering, content-based filtering, and embedding-based hybrid filtering.
- **Interactive UI**: User-friendly interface with components for comments, profiles, and home feed.
- **Data Processing**: Utilizes `pandas` and `numpy` for data manipulation.
- **Machine Learning Models**: Built with `pytorch` and `transformers` for recommendations.

---

## Prerequisites

### Backend
1. Python 3.7 or higher
2. Required libraries:
   - Flask
   - pandas
   - python-dotenv
   - transformers
   - numpy
   - pytorch
   - datasets
   - sklearn
   - scipy

### Frontend
1. Node.js (16.x or later recommended)
2. NPM (installed with Node.js)

---

## Installation

# Project Setup Instructions

## Backend Setup

1. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies


1. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencie
# Run the Flask server:

 ```bash
    python server.py


# Frontend Setup
Navigate to the frontend folder:

 ```bash
    cd WEBDEV

# Install dependencies:

 ```bash
    npm install

#  Start the React development server:

 ```bash
    npm start

# Project Structure

.
├── NCF_hybrid_recommendation.py
├── chatbotrecommender.ipynb
├── collaborative_filtering.py
├── content_based_filtering.py
├── embedding_based_hybrid_filtering.py
├── file_loader.py
├── WEBDEV
│   ├── package.json
│   ├── public
│   │   ├── index.html
│   │   └── manifest.json
│   ├── server
│   │   ├── server.py
│   │   ├── user_likes.xlsx
│   │   └── image_likes.xlsx
│   └── src
│       ├── App.js
│       ├── Components
│       │   ├── Home
│       │   │   ├── Homepage.js
│       │   │   └── Feedposts.js
│       ├── Pages
│       │   └── Home
│       │       ├── Home.js
│       │       └── Chatbot.jsx
│       └── index.js
└── model.ipynb






