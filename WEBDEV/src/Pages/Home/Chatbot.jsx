// // // // Chatbot.js
// // // import React, { useEffect, useState } from 'react';
// // // import { useNavigate } from 'react-router-dom';
// // // import Nav from '../../Components/Navigation/Nav';
// // // import Left from '../../Components/LeftSide/Left';
// // // import Post from './Prompt'; // Ensure the correct path
// // // import './chatbot.css';
// // // import './Home.css';

// // // const Chatbot = ({ setFriendsProfile }) => {
// // //   const [text, setText] = useState('');
// // //   const [image, setImage] = useState(null);
// // //   const [search, setSearch] = useState('');
// // //   const [showMenu, setShowMenu] = useState(false);
// // //   const [entries, setEntries] = useState([
// // //     {
// // //       prompt: 'Hi',
// // //       response: 'You can upload an image and ask me related recommendations via text or only text.',
// // //       images: [],
// // //       isExample: true,
// // //     },
// // //   ]);
// // //   const navigate = useNavigate();

// // //   useEffect(() => {
// // //     const token = localStorage.getItem('token');
// // //     const tokenExpiry = localStorage.getItem('tokenExpiry');

// // //     if (!token || !tokenExpiry) {
// // //       navigate('/');
// // //       return;
// // //     }

// // //     const currentTime = new Date().getTime();
// // //     if (currentTime >= parseInt(tokenExpiry, 10)) {
// // //       localStorage.removeItem('token');
// // //       localStorage.removeItem('tokenExpiry');
// // //       navigate('/');
// // //     }
// // //   }, [navigate]);

// // //   const handleImageUpload = (e) => {
// // //     if (e.target.files.length > 1) {
// // //       alert('Please upload only a single image!');
// // //       return;
// // //     }

// // //     const file = e.target.files[0];
// // //     if (file) {
// // //       setImage(file);
// // //     }
// // //   };

// // //   const handleSubmit = async () => {
// // //     if (!text.trim() && !image) {
// // //       alert('Please enter a message or upload an image.');
// // //       return;
// // //     }

// // //     const formData = new FormData();
// // //     formData.append('text', text);
// // //     if (image) {
// // //       formData.append('image', image);
// // //     }

// // //     try {
// // //       const response = await fetch('/chat', {
// // //         method: 'POST',
// // //         body: formData,
// // //       });

// // //       if (response.ok) {
// // //         const data = await response.json();
// // //         const newEntry = {
// // //           prompt: text,
// // //           response: data.response, // Assuming server returns { response: '...' }
// // //           images: data.images || [], // Assuming server returns { images: ['url1', 'url2', ...] }
// // //           isExample: false,
// // //         };
// // //         setEntries((prevEntries) => [newEntry, ...prevEntries]);
// // //         setText('');
// // //         setImage(null);
// // //       } else {
// // //         alert('Failed to send message.');
// // //       }
// // //     } catch (error) {
// // //       console.error('Error:', error);
// // //       alert('Error sending message.');
// // //     }
// // //   };

// // //   return (
// // //     <div className='interface'>
// // //       <Nav 
// // //         search={search}
// // //         setSearch={setSearch}
// // //         showMenu={showMenu}
// // //         setShowMenu={setShowMenu}
// // //       />

// // //       <div className="home">
// // //         <Left />

// // //         {/* Input Section */}
// // //         <div style={{display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'start'}}>
// // //         <div className='chat-input-container'>
// // //           <textarea
// // //             className='chat-text-input'
// // //             value={text}
// // //             onChange={(e) => setText(e.target.value)}
// // //             placeholder='Type your message...'
// // //           />
// // //           <div className="chat-input-actions" style={{marginTop:'-60px',marginRight:'30px'}}>
// // //             <label htmlFor='image-upload' className='chat-image-upload-label'>
// // //               <span role="img" aria-label="camera">📷</span>
// // //             </label>
// // //             <input
// // //               id='image-upload'
// // //               type='file'
// // //               accept='image/*'
// // //               className='chat-image-upload-input'
// // //               onChange={handleImageUpload}
// // //             />
// // //             <button className='chat-send-button' onClick={handleSubmit}>
// // //               <span role="img" aria-label="send">🚀</span>
// // //             </button>
// // //           </div>
// // //         </div>

// // //         {/* Entries Section */}
// // //         <div className="entries-container">
// // //           {entries.map((entry, index) => (
// // //             <Post
// // //               key={index}
// // //               prompt={entry.prompt}
// // //               response={entry.response}
// // //               images={entry.images}
// // //               isExample={entry.isExample}
// // //             />
// // //           ))}
// // //         </div>
// // //         </div>
        
// // //       </div>
// // //     </div>
// // //   );
// // // };

// // // export default Chatbot;


// // // Chatbot.js
// // import React, { useEffect, useState } from 'react';
// // import { useNavigate } from 'react-router-dom';
// // import Nav from '../../Components/Navigation/Nav';
// // import Left from '../../Components/LeftSide/Left';
// // import Post from './Prompt'; // Ensure the correct path
// // import './chatbot.css';
// // import './Home.css';

// // const Chatbot = ({ setFriendsProfile }) => {
// //   const [text, setText] = useState('');
// //   const [image, setImage] = useState(null);
// //   const [search, setSearch] = useState('');
// //   const [showMenu, setShowMenu] = useState(false);
// //   const [entries, setEntries] = useState([
// //     {
// //       prompt: 'Hi',
// //       response: 'You can upload an image and ask me related recommendations via text or only text.',
// //       recommendations: [], // Updated to include recommendations
// //       isExample: true,
// //     },
// //   ]);
// //   const navigate = useNavigate();

// //   useEffect(() => {
// //     const token = localStorage.getItem('token');
// //     const tokenExpiry = localStorage.getItem('tokenExpiry');

// //     if (!token || !tokenExpiry) {
// //       navigate('/');
// //       return;
// //     }

// //     const currentTime = new Date().getTime();
// //     if (currentTime >= parseInt(tokenExpiry, 10)) {
// //       localStorage.removeItem('token');
// //       localStorage.removeItem('tokenExpiry');
// //       navigate('/');
// //     }
// //   }, [navigate]);

// //   const handleImageUpload = (e) => {
// //     if (e.target.files.length > 1) {
// //       alert('Please upload only a single image!');
// //       return;
// //     }

// //     const file = e.target.files[0];
// //     if (file) {
// //       setImage(file);
// //     }
// //   };

// //   const handleSubmit = async () => {
// //     if (!text.trim() && !image) {
// //       alert('Please enter a message or upload an image.');
// //       return;
// //     }

// //     const formData = new FormData();
// //     formData.append('text', text);
// //     if (image) {
// //       formData.append('image', image);
// //     }

// //     try {
// //       const token = localStorage.getItem('token');
// //       const response = await fetch('http://localhost:5000/chat', {
// //         method: 'POST',
// //         headers: {
// //           'Authorization': `Bearer ${token}`, // Include JWT token
// //         },
// //         body: formData,
// //       });

// //       if (response.ok) {
// //         const data = await response.json();
// //         const newEntry = {
// //           prompt: text,
// //           response: data.response, // Server response message
// //           recommendations: data.recommendations || [], // Array of recommendations
// //           isExample: false,
// //         };
// //         setEntries((prevEntries) => [newEntry, ...prevEntries]);
// //         setText('');
// //         setImage(null);
// //       } else {
// //         const errorData = await response.json();
// //         alert(`Failed to send message: ${errorData.message || 'Unknown error.'}`);
// //       }
// //     } catch (error) {
// //       console.error('Error:', error);
// //       alert('Error sending message.');
// //     }
// //   };

// //   return (
// //     <div className='interface'>
// //       <Nav 
// //         search={search}
// //         setSearch={setSearch}
// //         showMenu={showMenu}
// //         setShowMenu={setShowMenu}
// //       />

// //       <div className="home">
// //         <Left />

// //         {/* Input Section */}
// //         <div style={{display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'start'}}>
// //           <div className='chat-input-container'>
// //             <textarea
// //               className='chat-text-input'
// //               value={text}
// //               onChange={(e) => setText(e.target.value)}
// //               placeholder='Type your message...'
// //             />
// //             <div className="chat-input-actions" style={{marginTop:'-60px',marginRight:'30px'}}>
// //               <label htmlFor='image-upload' className='chat-image-upload-label'>
// //                 <span role="img" aria-label="camera">📷</span>
// //               </label>
// //               <input
// //                 id='image-upload'
// //                 type='file'
// //                 accept='image/*'
// //                 className='chat-image-upload-input'
// //                 onChange={handleImageUpload}
// //               />
// //               <button className='chat-send-button' onClick={handleSubmit}>
// //                 <span role="img" aria-label="send">🚀</span>
// //               </button>
// //             </div>
// //           </div>

// //           {/* Entries Section */}
// //           <div className="entries-container">
// //             {entries.map((entry, index) => (
// //               <Post
// //                 key={index}
// //                 prompt={entry.prompt}
// //                 response={entry.response}
// //                 recommendations={entry.recommendations} // Pass recommendations
// //                 isExample={entry.isExample}
// //               />
// //             ))}
// //           </div>
// //         </div>
        
// //       </div>
// //     </div>
// //   );
// // };

// // export default Chatbot;


// // Chatbot.js
// import React, { useEffect, useState } from 'react';
// import { useNavigate } from 'react-router-dom';
// import Nav from '../../Components/Navigation/Nav';
// import Left from '../../Components/LeftSide/Left';
// import Post from './Prompt'; // Ensure the correct path
// import './chatbot.css';
// import './Home.css';

// const Chatbot = ({ setFriendsProfile }) => {
//   const [text, setText] = useState('');
//   const [image, setImage] = useState(null);
//   const [previewImage, setPreviewImage] = useState(null); // **Added**: State for image preview
//   const [search, setSearch] = useState('');
//   const [showMenu, setShowMenu] = useState(false);
//   const [entries, setEntries] = useState([
//     {
//       prompt: 'Hi',
//       response: 'You can upload an image and ask me related recommendations via text or only text.',
//       recommendations: [], // Updated to include recommendations
//       isExample: true,
//     },
//   ]);
//   const [isLoading, setIsLoading] = useState(false); // **Added**: State for loader
//   const navigate = useNavigate();

//   useEffect(() => {
//     const token = localStorage.getItem('token');
//     const tokenExpiry = localStorage.getItem('tokenExpiry');

//     if (!token || !tokenExpiry) {
//       navigate('/');
//       return;
//     }

//     const currentTime = new Date().getTime();
//     if (currentTime >= parseInt(tokenExpiry, 10)) {
//       localStorage.removeItem('token');
//       localStorage.removeItem('tokenExpiry');
//       navigate('/');
//     }
//   }, [navigate]);

//   const handleImageUpload = (e) => {
//     if (e.target.files.length > 1) {
//       alert('Please upload only a single image!');
//       return;
//     }

//     const file = e.target.files[0];
//     if (file) {
//       setImage(file);
//       setPreviewImage(URL.createObjectURL(file)); // **Added**: Set preview URL
//     }
//   };

//   const handleSubmit = async () => {
//     if (!text.trim() && !image) {
//       alert('Please enter a message or upload an image.');
//       return;
//     }

//     const formData = new FormData();
//     formData.append('text', text);
//     if (image) {
//       formData.append('image', image);
//     }

//     setIsLoading(true); // **Added**: Start loader

//     try {
//       const token = localStorage.getItem('token');
//       const response = await fetch('http://localhost:5000/chat', {
//         method: 'POST',
//         headers: {
//           'Authorization': `Bearer ${token}`, // Include JWT token
//         },
//         body: formData,
//       });

//       if (response.ok) {
//         const data = await response.json();
//         const newEntry = {
//           prompt: text,
//           response: data.response, // Server response message
//           recommendations: data.recommendations || [], // Array of recommendations
//           isExample: false,
//         };
//         setEntries((prevEntries) => [newEntry, ...prevEntries]);
//         setText('');
//         setImage(null);
//         setPreviewImage(null); // **Added**: Clear image preview after submission
//       } else {
//         const errorData = await response.json();
//         alert(`Failed to send message: ${errorData.message || 'Unknown error.'}`);
//       }
//     } catch (error) {
//       console.error('Error:', error);
//       alert('Error sending message.');
//     } finally {
//       setIsLoading(false); // **Added**: Stop loader
//     }
//   };

//   return (
//     <div className='interface'>
//       <Nav 
//         search={search}
//         setSearch={setSearch}
//         showMenu={showMenu}
//         setShowMenu={setShowMenu}
//       />

//       <div className="home">
//         <Left />

//         {/* Input Section */}
//         <div style={{display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'start'}}>
//           <div className='chat-input-container'>
//             <textarea
//               className='chat-text-input'
//               value={text}
//               onChange={(e) => setText(e.target.value)}
//               placeholder='Type your message...'
//             />
//             <div className="chat-input-actions" style={{marginTop:'-60px',marginRight:'30px'}}>
//               <label htmlFor='image-upload' className='chat-image-upload-label'>
//                 <span role="img" aria-label="camera">📷</span>
//               </label>
//               <input
//                 id='image-upload'
//                 type='file'
//                 accept='image/*'
//                 className='chat-image-upload-input'
//                 onChange={handleImageUpload}
//               />
//               <button className='chat-send-button' onClick={handleSubmit}>
//                 <span role="img" aria-label="send">🚀</span>
//               </button>
//             </div>
//           </div>

//           {/* **Added**: Image Preview */}
//           {previewImage && (
//             <div className="image-preview-container">
//               <img src={previewImage} alt="Preview" className="image-preview" />
//             </div>
//           )}

//           {/* **Added**: Loader */}
//           {isLoading && (
//             <div className="loader-container">
//               <div className="loader"></div>
//             </div>
//           )}

//           {/* Entries Section */}
//           <div className="entries-container">
//             {entries.map((entry, index) => (
//               <Post
//                 key={index}
//                 prompt={entry.prompt}
//                 response={entry.response}
//                 recommendations={entry.recommendations} // Pass recommendations
//                 isExample={entry.isExample}
//               />
//             ))}
//           </div>
//         </div>
        
//       </div>
//     </div>
//   );
// };

// export default Chatbot;


// Chatbot.js
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Nav from '../../Components/Navigation/Nav';
import Left from '../../Components/LeftSide/Left';
import Post from './Prompt'; // Ensure the correct path
import './chatbot.css';
import './Home.css';

const Chatbot = ({ setFriendsProfile }) => {
  const [text, setText] = useState('');
  const [image, setImage] = useState(null);
  const [previewImage, setPreviewImage] = useState(null); // **Added**: State for image preview
  const [search, setSearch] = useState('');
  const [showMenu, setShowMenu] = useState(false);
  const [entries, setEntries] = useState([
    {
      prompt: 'Hi',
      response: 'You can upload an image and ask me related recommendations via text or only text.',
      recommendations: [], // Updated to include recommendations
      isExample: true,
      previewImage: null, // **Added**: Initial previewImage as null
    },
  ]);
  const [isLoading, setIsLoading] = useState(false); // **Added**: State for loader
  const navigate = useNavigate();

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

  // **Added**: Cleanup for object URL to prevent memory leaks
  useEffect(() => {
    return () => {
      if (previewImage) {
        URL.revokeObjectURL(previewImage);
      }
    };
  }, [previewImage]);

  const handleImageUpload = (e) => {
    if (e.target.files.length > 1) {
      alert('Please upload only a single image!');
      return;
    }

    const file = e.target.files[0];
    if (file) {
      setImage(file);
      const previewURL = URL.createObjectURL(file);
      setPreviewImage(previewURL); // **Added**: Set preview URL
    }
  };

  const handleSubmit = async () => {
    if (!text.trim() && !image) {
      alert('Please enter a message or upload an image.');
      return;
    }

    const formData = new FormData();
    formData.append('text', text);
    if (image) {
      formData.append('image', image);
    }

    setIsLoading(true); // **Added**: Start loader

    try {
      const token = localStorage.getItem('token');
      const response = await fetch('http://localhost:5000/chat', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`, // Include JWT token
        },
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        const newEntry = {
          prompt: text,
          response: data.response, // Server response message
          recommendations: data.recommendations || [], // Array of recommendations
          isExample: false,
          previewImage: previewImage, // **Added**: Include previewImage in the entry
        };
        setEntries((prevEntries) => [newEntry, ...prevEntries]);
        setText('');
        setImage(null);
        setPreviewImage(null); // **Added**: Clear image preview after submission
      } else {
        const errorData = await response.json();
        alert(`Failed to send message: ${errorData.message || 'Unknown error.'}`);
      }
    } catch (error) {
      console.error('Error:', error);
      alert('Error sending message.');
    } finally {
      setIsLoading(false); // **Added**: Stop loader
    }
  };

  return (
    <div className='interface'>
      <Nav 
        search={search}
        setSearch={setSearch}
        showMenu={showMenu}
        setShowMenu={setShowMenu}
      />

      <div className="home">
        <Left />

        {/* Input Section */}
        <div style={{display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'start'}}>
          <div className='chat-input-container'>
            <textarea
              className='chat-text-input'
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder='Type your message...'
            />
            <div className="chat-input-actions" style={{marginTop:'-60px',marginRight:'30px'}}>
              <label htmlFor='image-upload' className='chat-image-upload-label'>
                <span role="img" aria-label="camera">📷</span>
              </label>
              <input
                id='image-upload'
                type='file'
                accept='image/*'
                className='chat-image-upload-input'
                onChange={handleImageUpload}
              />
              <button className='chat-send-button' onClick={handleSubmit} disabled={isLoading}>
                <span role="img" aria-label="send">🚀</span>
              </button>
            </div>
          </div>

          {/* **Added**: Image Preview */}
          {previewImage && (
            <div className="image-preview-container">
              <img src={previewImage} alt="Preview" className="image-preview" />
            </div>
          )}

          {/* **Added**: Loader */}
          {isLoading && (
            <div className="loader-container">
              <div className="loader"></div>
            </div>
          )}

          {/* Entries Section */}
          <div className="entries-container">
            {entries.map((entry, index) => (
              <Post
                key={index}
                prompt={entry.prompt}
                response={entry.response}
                recommendations={entry.recommendations} // Pass recommendations
                isExample={entry.isExample}
                previewImage={entry.previewImage} // **Added**: Pass previewImage to Post
              />
            ))}
          </div>
        </div>
        
      </div>
    </div>
  );
};

export default Chatbot;
