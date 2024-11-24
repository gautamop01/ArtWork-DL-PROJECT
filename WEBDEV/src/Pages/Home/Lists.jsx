import React, { useEffect, useState } from 'react';
import "../Home/Home.css";
import Left from "../../Components/LeftSide/Left";
import Nav from '../../Components/Navigation/Nav';
import { AiOutlineFilter } from 'react-icons/ai';
import "./Lists.css";
import { RotatingLines } from 'react-loader-spinner'; // Add a loader package (e.g., react-loader-spinner)
import { useNavigate } from 'react-router-dom';


const Home1 = ({ setFriendsProfile }) => {
  const [search, setSearch] = useState('');
  const [showMenu, setShowMenu] = useState(false);
  const [showFilter, setShowFilter] = useState(false);
  const [selectedFilters, setSelectedFilters] = useState({
    genre: [],
    style: [],
    artist: [],
  });
  const [images, setImages] = useState([]);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [hasMore, setHasMore] = useState(true);

  const navigate = useNavigate();

  const filterOptions = {
    genre: ["Landscape", "Portrait", "Abstract", "Still Life"],
    style: ["Impressionism", "Cubism", "Surrealism", "Realism"],
    artist: ["Vincent van Gogh", "Pablo Picasso", "Claude Monet", "Salvador Dalí"],
  };

  useEffect(() => {
    const token = localStorage.getItem('token');
    const tokenExpiry = localStorage.getItem('tokenExpiry');

    if (!token || !tokenExpiry) {
      navigate('/');
      return;
    }

    const currentTime = new Date().getTime();
    if (currentTime >= parseInt(tokenExpiry, 10)) {
      localStorage.removeItem('token');
      localStorage.removeItem('tokenExpiry');
      navigate('/');
    }
  }, [navigate]);

  const handleToggle = (filterType, option) => {
    setSelectedFilters((prev) => {
      const current = prev[filterType];
      const isSelected = current.includes(option);

      return {
        ...prev,
        [filterType]: isSelected
          ? current.filter((item) => item !== option)
          : [...current, option],
      };
    });
  };

  useEffect(() => {
    const fetchInitialImages = async () => {
      setLoading(true);
      try {
        const token = localStorage.getItem('token');
        const response = await fetch('http://localhost:5000/filter-images', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`,
          },
          body: JSON.stringify({
            filters: {}, // No filters for initial load
            page: 1,
          }),
        });
        const data = await response.json();
        if (data.images) {
          setImages(data.images);
          setHasMore(data.hasMore);
        }
      } catch (error) {
        console.error('Error fetching initial images:', error);
      }
      setLoading(false);
    };
  
    fetchInitialImages();
  }, []);
  

  const handleApplyFilters = async () => {
    setLoading(true);
    setPage(1);
    setImages([]);
    try {
      const token = localStorage.getItem('token');
      const response = await fetch('http://localhost:5000/filter-images', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({
          filters: selectedFilters,
          page: 1,
        }),
      });
      const data = await response.json();
      if (data.images) {
        setImages(data.images);
        setHasMore(data.hasMore);
      }
    } catch (error) {
      console.error('Error fetching filtered images:', error);
    }
    setLoading(false);
    setShowFilter(false);
  };

  const loadMoreImages = async () => {
    if (loading || !hasMore) return;

    setLoading(true);
    try {
      const token = localStorage.getItem('token');
      const response = await fetch('http://localhost:5000/filter-images', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({
          filters: selectedFilters,
          page: page + 1,
        }),
      });
      const data = await response.json();
      if (data.images) {
        setImages((prev) => [...prev, ...data.images]);
        setHasMore(data.hasMore);
        setPage((prev) => prev + 1);
      }
    } catch (error) {
      console.error('Error loading more images:', error);
    }
    setLoading(false);
  };

  useEffect(() => {
    const handleScroll = () => {
      if (
        window.innerHeight + document.documentElement.scrollTop >=
        document.documentElement.scrollHeight - 50
      ) {
        loadMoreImages();
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, [images, hasMore]);

  return (
    <div className="interface">
      <Nav
        search={search}
        setSearch={setSearch}
        showMenu={showMenu}
        setShowMenu={setShowMenu}
      />

      <div className="home">
        <Left />

        <div className="filter-fab" onClick={() => setShowFilter(!showFilter)}>
          <AiOutlineFilter size={24} />
        </div>

        {showFilter && (
          <div className="filter-sidebar">
            <h3>Filters</h3>
            {Object.keys(filterOptions).map((filterType) => (
              <div key={filterType} className="filter-section">
                <h4>{filterType.charAt(0).toUpperCase() + filterType.slice(1)}</h4>
                <ul>
                  {filterOptions[filterType].map((option) => (
                    <li key={option}>
                      <label>
                        <input
                          type="checkbox"
                          checked={selectedFilters[filterType].includes(option)}
                          onChange={() => handleToggle(filterType, option)}
                        />
                        {option}
                      </label>
                    </li>
                  ))}
                </ul>
              </div>
            ))}

            <div className="filter-actions">
              <button onClick={() => setShowFilter(false)}>Cancel</button>
              <button onClick={handleApplyFilters}>Apply</button>
            </div>
          </div>
        )}

        <div className="image-grid">
          {images.map((img, index) => (
            <div key={index} className="image-container">
              <img src={`data:image/jpeg;base64,${img}`} alt={`Art ${index}`} />
            </div>
          ))}

          {loading && (
            <div className="loading">
              <RotatingLines
                strokeColor="blue"
                strokeWidth="5"
                animationDuration="0.75"
                width="50"
                visible={true}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Home1;
