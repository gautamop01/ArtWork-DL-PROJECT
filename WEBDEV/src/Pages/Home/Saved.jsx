// frontend/src/pages/Home/LikedPhotos.js

import React, { useEffect, useState } from 'react';
import axios from 'axios';
import "../Home/Home.css"; // Ensure appropriate styling

import Left from "../../Components/LeftSide/Left";
import Nav from '../../Components/Navigation/Nav';
import { useNavigate } from 'react-router-dom';
import Post from '../../Components/Home/Post'; // Ensure the path is correct

const LikedPhotos = ({ setFriendsProfile }) => {
  const [likedPosts, setLikedPosts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  // Authentication Check
  useEffect(() => {
    const token = localStorage.getItem('token');
    const tokenExpiry = localStorage.getItem('tokenExpiry');

    if (!token || !tokenExpiry) {
      // If token or expiry is missing, redirect to login
      navigate('/');
      return;
    }

    const currentTime = new Date().getTime();
    if (currentTime >= parseInt(tokenExpiry, 10)) {
      // If token has expired, clear storage and redirect to login
      localStorage.removeItem('token');
      localStorage.removeItem('tokenExpiry');
      navigate('/');
    }
  }, [navigate]);

  // Fetch liked images on component mount
  useEffect(() => {
    fetchLikedImages();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const fetchLikedImages = async () => {
    const token = localStorage.getItem('token');
    try {
      const response = await axios.get('http://localhost:5000/get_all_liked', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.data.liked_images) {
        setLikedPosts(response.data.liked_images);
      } else {
        setLikedPosts([]);
      }
    } catch (err) {
      console.error('Error fetching liked images:', err.response?.data || err.message);
      setError('Failed to load liked images.');
    } finally {
      setLoading(false);
    }
  };

  const handleLike = async (imageIndex) => {
    const token = localStorage.getItem('token');
    try {
      await axios.post('http://localhost:5000/like-image', { image_index: imageIndex }, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      // Optionally, you can remove the liked image from the list or provide feedback
      // For simplicity, we'll refetch the liked images
      fetchLikedImages();
    } catch (error) {
      console.error('Error liking image:', error.response?.data || error.message);
      // Optionally, inform the user about the error
    }
  };

  return (
    <div className='interface'>
      <Nav 
        search={''} // If needed, adjust search functionality
        setSearch={() => {}} // If needed, adjust search functionality
        showMenu={false} // Adjust as necessary
        setShowMenu={() => {}} // Adjust as necessary
      />

      <div className="home">
        <Left />

        <div className="middle">
          <h2>Your Liked Photos</h2>

          {loading ? (
            <h4>Loading your liked photos...</h4>
          ) : error ? (
            <p style={{ color: 'red' }}>{error}</p>
          ) : likedPosts.length === 0 ? (
            <p>You haven't liked any photos yet.</p>
          ) : (
            <div className="posts-grid">
              {likedPosts.map(post => (
                <Post 
                  key={`${post.index}-${post.timestamp}`} // Ensure uniqueness
                  artist={post.artist}
                  imageIndex={post.index}
                  description={`${post.genre}, ${post.style}`}
                  image={post.image}
                  onLike={() => handleLike(post.index)}
                  setFriendsProfile={setFriendsProfile}
                  timestamp={post.timestamp} // Ensure this field exists
                />
              ))}
            </div>
          )}
        </div>

        {/* <Right /> */}
      </div>
    </div>
  );
}

export default LikedPhotos;
