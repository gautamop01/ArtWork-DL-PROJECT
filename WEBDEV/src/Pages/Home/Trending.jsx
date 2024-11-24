// frontend/src/pages/Home/Trending.js

import React, { useEffect, useState } from 'react';
import axios from 'axios';
import "../Home/Home.css"; // Ensure appropriate styling

import Left from "../../Components/LeftSide/Left";
import Nav from '../../Components/Navigation/Nav';
import { useNavigate } from 'react-router-dom';
import Post from '../../Components/Home/Post'; // Ensure the path is correct

const Trending = ({ setFriendsProfile }) => {
  const [trendingPosts, setTrendingPosts] = useState([]);
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

  // Fetch trending images on component mount
  useEffect(() => {
    fetchTrendingImages();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const fetchTrendingImages = async () => {
    const token = localStorage.getItem('token');
    try {
      const response = await axios.get('http://localhost:5000/trending', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.data.trending_images) {
        setTrendingPosts(response.data.trending_images);
      } else {
        setTrendingPosts([]);
      }
    } catch (err) {
      console.error('Error fetching trending images:', err.response?.data || err.message);
      setError('Failed to load trending images.');
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

      // Optionally, you can provide feedback or update the like count locally
      // For simplicity, we'll refetch the trending images to update like counts
      fetchTrendingImages();
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
          <h2>Trending Images</h2>

          {loading ? (
            <h4>Loading trending images...</h4>
          ) : error ? (
            <p style={{ color: 'red' }}>{error}</p>
          ) : trendingPosts.length === 0 ? (
            <p>No trending images available at the moment.</p>
          ) : (
            <div className="posts-grid">
              {trendingPosts.map(post => (
                <Post 
                  key={`${post.index}-${post.like_count}`} // Ensure uniqueness
                  artist={post.artist}
                  imageIndex={post.index}
                  description={`${post.genre}, ${post.style}`}
                  image={post.image}
                  onLike={() => handleLike(post.index)}
                  setFriendsProfile={setFriendsProfile}
                  likeCount={post.like_count} // Pass like count to the Post component
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

export default Trending;
