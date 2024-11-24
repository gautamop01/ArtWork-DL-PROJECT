// // import React, { useState } from 'react';
// // import "./Prompt.css";
// // import ChatbotLogo from "./logo.png"; // Chatbot Logo
// // import AiLogo from "./logo.png"; // AI Logo beside response
// // import { AiOutlineClose, AiOutlineLike, AiOutlineSave } from "react-icons/ai";
// // import Slider from "react-slick";
// // import "slick-carousel/slick/slick.css"; 
// // import "slick-carousel/slick/slick-theme.css";

// // const Post = ({ key, prompt, response, images }) => {
// //   const [showModal, setShowModal] = useState(false);
// //   const [selectedPhoto, setSelectedPhoto] = useState(null);

// //   // Slider settings
// //   const sliderSettings = {
// //     dots: true,
// //     infinite: true,
// //     speed: 600,
// //     slidesToShow: 3,
// //     slidesToScroll: 1,
// //     autoplay: true,
// //     autoplaySpeed: 3000,
// //     centerMode: true,
// //     centerPadding: '0px',
// //     arrows: false,
// //     responsive: [
// //       {
// //         breakpoint: 1024,
// //         settings: {
// //           slidesToShow: 2,
// //         },
// //       },
// //       {
// //         breakpoint: 600,
// //         settings: {
// //           slidesToShow: 1,
// //         },
// //       },
// //     ],
// //   };

// //   // Open modal to view photo
// //   const openModal = (photo) => {
// //     setSelectedPhoto(photo);
// //     setShowModal(true);
// //   };

// //   // Close modal
// //   const closeModal = () => {
// //     setShowModal(false);
// //     setSelectedPhoto(null);
// //   };

// //   return (
// //     <div key={key} className="chatbot-container1">
// //       {/* Interaction Section */}
// //       <div className="interaction-section">
// //         {/* Prompt Asked */}
// //         <div className="prompt-display">
// //           <p><span className="prompt-label">Prompt:</span> {prompt}</p>
// //         </div>

// //         {/* Response */}
// //         <div className="response-display">
// //           <img src={AiLogo} alt="AI Logo" className="ai-logo animated-ai-logo" />
// //           <p><span className="response-label">Response:</span> {response}</p>
// //         </div>
// //       </div>

// //       {/* Photo Slider */}
// //       <div className="photo-slider">
// //         <Slider {...sliderSettings}>
// //           {images.map((photo, index) => (
// //             <div 
// //               key={index} 
// //               className="slider-item" 
// //               onClick={() => openModal(photo)}
// //             >
// //               <img src={photo} alt={`Chatbot response ${index + 1}`} className="slider-image" />
// //             </div>
// //           ))}
// //         </Slider>
// //       </div>

// //       {/* Modal View */}
// //       {showModal && (
// //         <div className="modal-overlay" onClick={closeModal}>
// //           <div className="modal-content" onClick={(e) => e.stopPropagation()}>
// //             <AiOutlineClose className="modal-close" onClick={closeModal} />
// //             <div className="modal-image-container">
// //               <img src={selectedPhoto} alt="Selected" className="modal-photo" />
// //               <div className="modal-actions">
// //                 <AiOutlineLike className="action-icon" title="Like" onClick={() => alert('Liked!')} />
// //                 <AiOutlineSave className="action-icon" title="Save" onClick={() => alert('Saved!')} />
// //               </div>
// //             </div>
// //           </div>
// //         </div>
// //       )}
// //     </div>
// //   );
// // };

// // export default Post;


// // Post.js
// import React, { useState } from 'react';
// import "./Prompt.css";
// import ChatbotLogo from "./logo.png"; // Chatbot Logo
// import AiLogo from "./logo.png"; // AI Logo beside response
// import { AiOutlineClose, AiOutlineLike, AiOutlineSave } from "react-icons/ai";
// import Slider from "react-slick";
// import "slick-carousel/slick/slick.css"; 
// import "slick-carousel/slick/slick-theme.css";

// const Post = ({ prompt, response, recommendations, isExample }) => {
//   const [showModal, setShowModal] = useState(false);
//   const [selectedPhoto, setSelectedPhoto] = useState(null);
//   const [selectedMetadata, setSelectedMetadata] = useState(null); // To store metadata

//   // Slider settings
//   const sliderSettings = {
//     dots: true,
//     infinite: true,
//     speed: 600,
//     slidesToShow: 3,
//     slidesToScroll: 1,
//     autoplay: true,
//     autoplaySpeed: 3000,
//     centerMode: true,
//     centerPadding: '0px',
//     arrows: false,
//     responsive: [
//       {
//         breakpoint: 1024,
//         settings: {
//           slidesToShow: 2,
//         },
//       },
//       {
//         breakpoint: 600,
//         settings: {
//           slidesToShow: 1,
//         },
//       },
//     ],
//   };

//   // Open modal to view photo and metadata
//   const openModal = (photo, metadata) => {
//     setSelectedPhoto(photo);
//     setSelectedMetadata(metadata);
//     setShowModal(true);
//   };

//   // Close modal
//   const closeModal = () => {
//     setShowModal(false);
//     setSelectedPhoto(null);
//     setSelectedMetadata(null);
//   };

//   // Function to format Base64 image
//   const formatBase64Image = (base64String) => {
//     return `data:image/jpeg;base64,${base64String}`;
//   };

//   return (
//     <div className="chatbot-container1">
//       {/* Interaction Section */}
//       <div className="interaction-section">
//         {/* Prompt Asked */}
//         <div className="prompt-display">
//           <p><span className="prompt-label">Prompt:</span> {prompt}</p>
//         </div>

//         {/* Response */}
//         <div className="response-display">
//           <img src={AiLogo} alt="AI Logo" className="ai-logo animated-ai-logo" />
//           <p><span className="response-label">Response:</span> {response}</p>
//         </div>
//       </div>

//       {/* Recommendations Slider */}
//       {recommendations && recommendations.length > 0 && (
//         <div className="photo-slider">
//           <Slider {...sliderSettings}>
//             {recommendations.map((rec, index) => (
//               <div 
//                 key={index} 
//                 className="slider-item" 
//                 onClick={() => openModal(rec.image, rec)}
//               >
//                 <img 
//                   src={formatBase64Image(rec.image)} 
//                   alt={`Recommendation ${index + 1}`} 
//                   className="slider-image" 
//                 />
//               </div>
//             ))}
//           </Slider>
//         </div>
//       )}

//       {/* Modal View */}
//       {showModal && selectedPhoto && selectedMetadata && (
//         <div className="modal-overlay" onClick={closeModal}>
//           <div className="modal-content" onClick={(e) => e.stopPropagation()}>
//             <AiOutlineClose className="modal-close" onClick={closeModal} style={{zIndex:999}} />
//             <div className="modal-image-container">
//               <img src={formatBase64Image(selectedPhoto)} alt="Selected" className="modal-photo" />
//               {/* Metadata Overlay */}
//               <div className="metadata-overlay">
//                 <p><strong>Artist:</strong> {selectedMetadata.artist}</p>
//                 <p><strong>Style:</strong> {selectedMetadata.style}</p>
//                 <p><strong>Genre:</strong> {selectedMetadata.genre}</p>
//               </div>
//               <div className="modal-actions">
//                 <AiOutlineLike className="action-icon" title="Like" onClick={() => alert('Liked!')} />
//                 <AiOutlineSave className="action-icon" title="Save" onClick={() => alert('Saved!')} />
//               </div>
//             </div>
//           </div>
//         </div>
//       )}
//     </div>
//   );
// };

// export default Post;


// Post.js
import React, { useState } from 'react';
import "./Prompt.css";
import ChatbotLogo from "./logo.png"; // Chatbot Logo
import AiLogo from "./logo.png"; // AI Logo beside response
import { AiOutlineClose, AiOutlineLike, AiOutlineSave } from "react-icons/ai";
import Slider from "react-slick";
import "slick-carousel/slick/slick.css"; 
import "slick-carousel/slick/slick-theme.css";

const Post = ({ prompt, response, recommendations, isExample, previewImage }) => { // **Added**: previewImage prop
  const [showModal, setShowModal] = useState(false);
  const [selectedPhoto, setSelectedPhoto] = useState(null);
  const [selectedMetadata, setSelectedMetadata] = useState(null); // To store metadata

  // Slider settings
  const sliderSettings = {
    dots: true,
    infinite: true,
    speed: 600,
    slidesToShow: Math.min(3, recommendations.length), // Adjust based on number of recommendations
    slidesToScroll: 1,
    autoplay: true,
    autoplaySpeed: 3000,
    centerMode: true,
    centerPadding: '0px',
    arrows: false,
    responsive: [
      {
        breakpoint: 1024,
        settings: {
          slidesToShow: Math.min(2, recommendations.length),
        },
      },
      {
        breakpoint: 600,
        settings: {
          slidesToShow: 1,
        },
      },
    ],
  };

  // Open modal to view photo and metadata
  const openModal = (photo, metadata) => {
    setSelectedPhoto(photo);
    setSelectedMetadata(metadata);
    setShowModal(true);
  };

  // Close modal
  const closeModal = () => {
    setShowModal(false);
    setSelectedPhoto(null);
    setSelectedMetadata(null);
  };

  // Function to format Base64 image
  const formatBase64Image = (base64String) => {
    return `data:image/jpeg;base64,${base64String}`;
  };

  return (
    <div className="chatbot-container1">
      {/* Interaction Section */}
      <div className="interaction-section">
        {/* **Added**: Display Preview Image */}
        {previewImage && (
          <div className="preview-image-container">
            <img src={previewImage} alt="Preview" className="preview-image" />
          </div>
        )}

        {/* Prompt Asked */}
        <div className="prompt-display">
          <p><span className="prompt-label">Prompt:</span> {prompt}</p>
        </div>

        {/* Response */}
        <div className="response-display">
          <img src={AiLogo} alt="AI Logo" className="ai-logo animated-ai-logo" />
          <p><span className="response-label">Response:</span> {response}</p>
        </div>
      </div>

      {/* Recommendations Slider */}
      {recommendations && recommendations.length > 0 && (
        <div className="photo-slider">
          <Slider {...sliderSettings}>
            {recommendations.map((rec, index) => (
              <div 
                key={index} 
                className="slider-item" 
                onClick={() => openModal(rec.image, rec)}
              >
                {rec.image ? (
                  <img 
                    src={formatBase64Image(rec.image)} 
                    alt={`Recommendation ${index + 1}`} 
                    className="slider-image" 
                  />
                ) : (
                  <div className="no-image-placeholder">No Image Available</div>
                )}
              </div>
            ))}
          </Slider>
        </div>
      )}

      {/* Modal View */}
      {showModal && selectedPhoto && selectedMetadata && (
        <div className="modal-overlay" onClick={closeModal}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <AiOutlineClose className="modal-close" onClick={closeModal} style={{zIndex:999}} />
            <div className="modal-image-container">
              {selectedPhoto ? (
                <img src={formatBase64Image(selectedPhoto)} alt="Selected" className="modal-photo" />
              ) : (
                <div className="no-image-placeholder">No Image Available</div>
              )}
              {/* Metadata Overlay */}
              <div className="metadata-overlay">
                <p><strong>Artist:</strong> {selectedMetadata.artist}</p>
                <p><strong>Style:</strong> {selectedMetadata.style}</p>
                <p><strong>Genre:</strong> {selectedMetadata.genre}</p>
              </div>
              <div className="modal-actions">
                <AiOutlineLike className="action-icon" title="Like" onClick={() => alert('Liked!')} />
                <AiOutlineSave className="action-icon" title="Save" onClick={() => alert('Saved!')} />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Post;
