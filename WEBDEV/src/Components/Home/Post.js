// frontend/src/Components/Post/Post.jsx

import React, { useState } from 'react';
import "./Post.css";
import FavoriteBorderOutlinedIcon from '@mui/icons-material/FavoriteBorderOutlined';
import FavoriteRoundedIcon from '@mui/icons-material/FavoriteRounded';
import MoreVertRoundedIcon from '@mui/icons-material/MoreVertRounded';
import Censor from "./censor.jpg"
import { PiSmileySad } from "react-icons/pi";
import { IoVolumeMuteOutline } from "react-icons/io5";
import { MdBlockFlipped } from "react-icons/md";
import { AiOutlineDelete } from "react-icons/ai";
import { MdReportGmailerrorred } from "react-icons/md";

import { LiaFacebookF } from "react-icons/lia";
import { FiInstagram } from "react-icons/fi";
import { BiLogoLinkedin } from "react-icons/bi";
import { AiFillYoutube } from "react-icons/ai";
import { RxTwitterLogo } from "react-icons/rx";
import { FiGithub } from "react-icons/fi";

import { Link } from 'react-router-dom';

const Post = ({ artist, imageIndex, description, image, onLike, setFriendsProfile, timestamp,iscensored }) => {
  const [like, setLike] = useState(0); // Initialize like count as 0
  const [unlike, setUnlike] = useState(false);
  const [filledLike, setFilledLike] = useState(<FavoriteBorderOutlinedIcon />);
  const [unFilledLike, setUnFilledLike] = useState(false);

  const [showOptions, setShowOptions] = useState(false);
  const [showSocialIcons, setShowSocialIcons] = useState(false);

  const handleLikes = () => {
    setLike(unlike ? like - 1 : like + 1);
    setUnlike(!unlike);

    setFilledLike(unFilledLike ? <FavoriteBorderOutlinedIcon /> : <FavoriteRoundedIcon />);
    setUnFilledLike(!unFilledLike);

    // Notify parent component about the like action
    onLike(imageIndex);
  };

  const handleDelete = () => {
    // Implement delete functionality if needed
    setShowOptions(false);
  };

  const toggleOptions = () => {
    setShowOptions(!showOptions);
  };

  const toggleSocialIcons = () => {
    setShowSocialIcons(!showSocialIcons);
  };

  return (
    <div className='post'>
      {/* Post Header */}
      <div className='post-header'>
        <div className='post-info'>
          <h2>{artist}</h2>
          <span className='post-timestamp'>{timestamp}</span>
        </div>

        <div className='post-options'>
          {showOptions && (
            <div className="options">
              <button><PiSmileySad /> Explanation</button>
              <button><PiSmileySad /> Not Interested</button>
              <button><IoVolumeMuteOutline /> Mute User</button>
              <button><MdBlockFlipped /> Block User</button>
              <button onClick={handleDelete}><AiOutlineDelete /> Delete Post</button>
              <button><MdReportGmailerrorred /> Report</button>
            </div>
          )}
          <MoreVertRoundedIcon className='post-vertical-icon' onClick={toggleOptions} />
        </div>
      </div>

      {/* Post Body */}
      <p className='post-body'>{description}</p>

      {/* Post Image */}
      {image && (
        (iscensored)?<img src={Censor} className="post-image"></img>:<img src={`data:image/jpeg;base64,${image}`} alt="Artwork" className="post-image" />
      )}

      {/* Post Footer */}
      <div className="post-footer">
        <div className="like-share-section">
          <div className="like-icons">
            <span className='heart' onClick={handleLikes}>
              {filledLike}
            </span>
            <span className='like-count'>{like} Likes</span>
          </div>

          <div className="share-icons">
            <span className='share-icon' onClick={toggleSocialIcons}>
              Share
            </span>
            {showSocialIcons && (
              <div className="social-buttons">
                <a href="http://www.facebook.com" target="_blank" rel="noopener noreferrer" className="social-margin">
                  <div className="social-icon facebook">
                    <LiaFacebookF className='social-links' />
                  </div>
                </a>

                <a href="https://www.instagram.com/" target="_blank" rel="noopener noreferrer" className="social-margin">
                  <div className="social-icon instagram">
                    <FiInstagram className='social-links' />
                  </div>
                </a>

                <a href="http://linkedin.com/" target="_blank" rel="noopener noreferrer" className="social-margin">
                  <div className="social-icon linkedin">
                    <BiLogoLinkedin className='social-links' />
                  </div>
                </a>

                <a href="https://github.com/" target="_blank" rel="noopener noreferrer" className="social-margin">
                  <div className="social-icon github">
                    <FiGithub className='social-links' />
                  </div>
                </a>

                <a href="http://youtube.com/" target="_blank" rel="noopener noreferrer" className="social-margin">
                  <div className="social-icon youtube">
                    <AiFillYoutube className='social-links' />
                  </div>
                </a>

                <a href="http://twitter.com/" target="_blank" rel="noopener noreferrer" className="social-margin">
                  <div className="social-icon twitter">
                    <RxTwitterLogo className='social-links' />
                  </div>
                </a>
              </div>
            )}
          </div>
        </div>

        <div className="comment-section">
          <span className='comment-count'>0 Comments</span>
        </div>
      </div>
    </div>
  );
}

export default Post;
