// // // // // // // import {  useEffect, useState } from 'react'
// // // // // // // import Profile from "../../assets/profile.jpg"
// // // // // // // import img1 from "../../assets/Post Images/img1.jpg"
// // // // // // // import img2 from "../../assets/Post Images/img2.jpg"
// // // // // // // import img3 from "../../assets/Post Images/img3.jpg"
// // // // // // // import img4 from "../../assets/Post Images/img4.jpg"
// // // // // // // import img5 from "../../assets/Post Images/img5.jpg"
// // // // // // // import img6 from "../../assets/Post Images/img6.jpg"


// // // // // // // import DPimg1 from "../../assets/DP/img1.jpg"
// // // // // // // import DPimg2 from "../../assets/DP/img2.jpg"
// // // // // // // import DPimg3 from "../../assets/DP/img3.jpg"
// // // // // // // import DPimg4 from "../../assets/DP/img4.jpg"
// // // // // // // import DPimg5 from "../../assets/DP/img5.jpg"
// // // // // // // import DPimg6 from "../../assets/DP/img6.jpg"

// // // // // // // import cover from "../../assets/Info-Dp/img-3.jpg"

// // // // // // // import Cover1 from "../../assets/Friends-Cover/cover-1.jpg"
// // // // // // // import Cover2 from "../../assets/Friends-Cover/cover-2.jpg"
// // // // // // // import Cover3 from "../../assets/Friends-Cover/cover-3.jpg"
// // // // // // // import Cover5 from "../../assets/Friends-Cover/cover-5.jpg"
// // // // // // // import Cover7 from "../../assets/Friends-Cover/cover-7.jpg"
// // // // // // // import Cover8 from "../../assets/Friends-Cover/cover-8.jpg"
// // // // // // // import Cover9 from "../../assets/Friends-Cover/cover-9.jpg"

// // // // // // // import Uimg1 from "../../assets/User-post/img1.jpg"
// // // // // // // import Uimg2 from "../../assets/User-post/img2.jpg"
// // // // // // // import Uimg3 from "../../assets/User-post/img3.jpg"


// // // // // // // import "../Home/Home.css"

// // // // // // // import Left from "../../Components/LeftSide/Left"
// // // // // // // import Middle from "../../Components/MiddleSide/Middle"
// // // // // // // import Right from '../../Components/RightSide/Right'
// // // // // // // import Nav from '../../Components/Navigation/Nav'
// // // // // // // import moment from 'moment/moment'
// // // // // // // import { useNavigate } from 'react-router-dom'




// // // // // // // const Home = ({setFriendsProfile}) => {
  
// // // // // // //     const [posts,setPosts] = useState(
// // // // // // //         [
// // // // // // //           {
// // // // // // //             id:1,
// // // // // // //             username:"Harry",
// // // // // // //             profilepicture:DPimg1,
// // // // // // //             img:img1,
// // // // // // //             datetime:moment("20230131", "YYYYMMDD").fromNow(),
// // // // // // //             body:"My 1st Post, Have A Good Day Lorem ipsum dolor sit, amet consectetur adipisicing elit. Porro ipsum laborum necessitatibus ex doloragnam ea?",
// // // // // // //             like: 44,
// // // // // // //             comment:3,
// // // // // // //             unFilledLike:true,
// // // // // // //             coverpicture:Cover1,
// // // // // // //             userid:"@Iamharry",
// // // // // // //             ModelCountryName:"USA",
// // // // // // //             ModelJobName:"Java Developer",
// // // // // // //             ModelJoinedDate:"Joined in 2019-02-28",
// // // // // // //             followers:1478
// // // // // // //           },
// // // // // // //           {
// // // // // // //             id:2,
// // // // // // //             username:"chris dhaniel",
// // // // // // //             profilepicture:DPimg2,
// // // // // // //             img:img2,
// // // // // // //             datetime:moment("20230605", "YYYYMMDD").fromNow(),
// // // // // // //             body:"My 2st Post, Have A Bad Day Lorem ipsum dolor sit, amet consectetur adipisicing elit. Porro ipsum laborum necessitatibus ex dolor reiciendis, consequuntur placeat repellat magnam ea?",
// // // // // // //             like: 84,
// // // // // // //             comment:3,
// // // // // // //             coverpicture:Cover2,
// // // // // // //             userid:"@chris777",
// // // // // // //             ModelCountryName:"Australia",
// // // // // // //             ModelJobName:"Cyber Security",
// // // // // // //             ModelJoinedDate:"Joined in 2018-01-17",
// // // // // // //             followers:1730
// // // // // // //           },
// // // // // // //           {
// // // // // // //             id:3,
// // // // // // //             username:"April",
// // // // // // //             profilepicture:DPimg3,
// // // // // // //             img:img3,
// // // // // // //             datetime:moment("20230813", "YYYYMMDD").fromNow(),
// // // // // // //             body:"My 3st Post, Have A Nice Day Lorem ipsum dolor sit, amet consectetur adipisicing elit. Porro ipsum laborum necessitatibus ex dolor reiciendis, consequuntur",
// // // // // // //             like: 340,
// // // // // // //             comment:76,
// // // // // // //             coverpicture:Cover3,
// // // // // // //             userid:"@April",
// // // // // // //             ModelCountryName:"India",
// // // // // // //             ModelJobName:"Python Developer",
// // // // // // //             ModelJoinedDate:"Joined in 2022-03-01",
// // // // // // //             followers:426
// // // // // // //           },
// // // // // // //           {
// // // // // // //             id:4,
// // // // // // //             username:"GauTam",
// // // // // // //             profilepicture:Profile,
// // // // // // //             img:Uimg1,
// // // // // // //             datetime:moment("20230310", "YYYYMMDD").fromNow(),
// // // // // // //             body:"Lorem ipsum dolor sit amet consectetur adipisicing elit. Officia illum provident consequuntur reprehenderit tenetur, molestiae quae blanditiis rem placeat! Eligendi, qui quia quibusdam dolore molestiae veniam neque fuga explicabo illum?",
// // // // // // //             like: 22,
// // // // // // //             comment:3,
// // // // // // //             coverpicture:cover,
// // // // // // //             userid:"@gautamop01",
// // // // // // //             ModelCountryName:"India",
// // // // // // //             ModelJobName:"Web Developer ",
// // // // // // //             ModelJoinedDate:"Joined in 2023-08-12",
// // // // // // //             followers:5000
// // // // // // //           },
// // // // // // //           {
// // // // // // //             id:5,
// // // // // // //             username:"Lara",
// // // // // // //             profilepicture:DPimg4,
// // // // // // //             img:img4,
// // // // // // //             datetime:moment("20200101", "YYYYMMDD").fromNow(),
// // // // // // //             body:"My 4st Post, Have A Dull DayLorem ipsum dolor sit amet consectetur adipisicing elit. Iure veritatis numquam, ex explicabo tempore eum autem. Distinctio, odit fugiat rerum animi mollitia placeat? At ipsam debitis animi rem suscipit dicta dolor eveniet impedit minus. Quidem odit autem quia facere consectetur vero placeat delectus enim aspernatur",
// // // // // // //             like: 44,
// // // // // // //             comment:3,
// // // // // // //             coverpicture:Cover5,
// // // // // // //             userid:"@laralara",
// // // // // // //             ModelCountryName:"London",
// // // // // // //             ModelJobName:"CEO in Google",
// // // // // // //             ModelJoinedDate:"Joined in 2023-04-15",
// // // // // // //             followers:3005
// // // // // // //           },
// // // // // // //           {
// // // // // // //             id:6,
// // // // // // //             username:"GauTam",
// // // // // // //             profilepicture:Profile,
// // // // // // //             img:Uimg2,
// // // // // // //             datetime:moment("20230618", "YYYYMMDD").fromNow(),
// // // // // // //             body:"Lorem ipsum dolor sit amet consectetur adipisicing elit. Officia illum provident consequuntur reprehenderit tenetur, molestiae quae blanditiis rem placeat! Eligendi, qui quia quibusdam dolore molestiae veniam neque fuga explicabo illum?",
// // // // // // //             like: 84,
// // // // // // //             comment:3,
// // // // // // //             coverpicture:cover,
// // // // // // //             userid:"@gautamop01",
// // // // // // //             ModelCountryName:"India",
// // // // // // //             ModelJobName:"Web Developer ",
// // // // // // //             ModelJoinedDate:"Joined in 2023-08-12",
// // // // // // //             followers:5000
// // // // // // //           },
// // // // // // //           {
// // // // // // //             id:7,
// // // // // // //             username:"Kenny",
// // // // // // //             profilepicture:DPimg5,
// // // // // // //             img:img5,
// // // // // // //             datetime:moment("20230505", "YYYYMMDD").fromNow(),
// // // // // // //             body:"My 5st Post, Have A Awesome Day Lorem ipsum dolor sit, amet consectetur adipisicing elit. Porro ipsum laborum necessitatibus ex",
// // // // // // //             like: 30,
// // // // // // //             comment:3,
// // // // // // //             coverpicture:Cover7,
// // // // // // //             userid:"@kenny80",
// // // // // // //             ModelCountryName:"South Africa",
// // // // // // //             ModelJobName:"Full Stack Web Developer in Twitter",
// // // // // // //             ModelJoinedDate:"Joined in 2020-08-09",
// // // // // // //             followers:626
// // // // // // //           },
// // // // // // //           {
// // // // // // //             id:8,
// // // // // // //             username:"GauTam",
// // // // // // //             profilepicture:Profile,
// // // // // // //             img:Uimg3,
// // // // // // //             datetime:moment("20230219", "YYYYMMDD").fromNow(),
// // // // // // //             body:"Lorem ipsum dolor sit amet consectetur adipisicing elit. Officia illum provident consequuntur reprehenderit tenetur, molestiae quae blanditiis rem placeat! Eligendi, qui quia quibusdam dolore molestiae veniam neque fuga explicabo illum?",
// // // // // // //             like: 340,
// // // // // // //             comment:3,
// // // // // // //             coverpicture:Cover8,
// // // // // // //             userid:"@gautamop01",
// // // // // // //             ModelCountryName:"India",
// // // // // // //             ModelJobName:"Web Developer ",
// // // // // // //             ModelJoinedDate:"Joined in 2023-08-12",
// // // // // // //             followers:5000

// // // // // // //           },
// // // // // // //           {
// // // // // // //             id:9,
// // // // // // //             username:"Reyana",
// // // // // // //             profilepicture:DPimg6,
// // // // // // //             img:img6,
// // // // // // //             datetime:moment("20230404", "YYYYMMDD").fromNow(),
// // // // // // //             body:"My 6st Post, Have A Half Day Lorem ipsum dolor sit, amet consectetur adipisicing elit. Porro ipsum laborum necessitatibus ex dolor reiciendis, consequuntur",
// // // // // // //             like: 844,
// // // // // // //             comment:3,
// // // // // // //             coverpicture:Cover9,
// // // // // // //             userid:"@reyanaRey",
// // // // // // //             ModelCountryName:"Russia",
// // // // // // //             ModelJobName:"Back End Developer in Microsoft",
// // // // // // //             ModelJoinedDate:"Joined in 2020-02-29",
// // // // // // //             followers:3599
// // // // // // //            }
// // // // // // //         ]
// // // // // // //       )

// // // // // // //       const [body,setBody] =useState("")
// // // // // // //       const [importFile,setImportFile] =useState("")
      

// // // // // // //       const handleSubmit =(e)=>{
// // // // // // //         e.preventDefault()
        
        
// // // // // // //         const id =posts.length ? posts[posts.length -1].id +1 :1
// // // // // // //         const username="GauTam"
// // // // // // //         const profilepicture=Profile
// // // // // // //         const datetime=moment.utc(new Date(), 'yyyy/MM/dd kk:mm:ss').local().startOf('seconds').fromNow()
// // // // // // //         const img =images ? {img:URL.createObjectURL(images)} : null
        
// // // // // // //         const obj ={id:id,
// // // // // // //                    profilepicture:profilepicture,
// // // // // // //                    username:username,
// // // // // // //                    datetime:datetime,
// // // // // // //                    img:img && (img.img),
// // // // // // //                    body:body,
// // // // // // //                    like:0,
// // // // // // //                    comment:0
// // // // // // //                   }

        

// // // // // // //         const insert =[...posts,obj]
// // // // // // //         setPosts(insert)
// // // // // // //         setBody("")
// // // // // // //         setImages(null)

// // // // // // //       }
   
// // // // // // //    const [search,setSearch] =useState("")

    
// // // // // // //   const [following,setFollowing] =useState("")
        
// // // // // // //   const [showMenu,setShowMenu] =useState(false)
// // // // // // //   const [images,setImages] =  useState(null)
// // // // // // //   const navigate = useNavigate();

// // // // // // // useEffect(() => {
// // // // // // //   const token = localStorage.getItem('token');
// // // // // // //   const tokenExpiry = localStorage.getItem('tokenExpiry');

// // // // // // //   if (!token || !tokenExpiry) {
// // // // // // //       // If token or expiry is missing, redirect to login
// // // // // // //       navigate('/');
// // // // // // //       return;
// // // // // // //   }

// // // // // // //   const currentTime = new Date().getTime();
// // // // // // //   if (currentTime >= parseInt(tokenExpiry, 10)) {
// // // // // // //       // If token has expired, clear storage and redirect to login
// // // // // // //       localStorage.removeItem('token');
// // // // // // //       localStorage.removeItem('tokenExpiry');
// // // // // // //       navigate('/');
// // // // // // //   }
// // // // // // // }, [navigate]);

// // // // // // //   return (
// // // // // // //     <div className='interface'>
// // // // // // //         <Nav 
// // // // // // //         search={search}
// // // // // // //         setSearch={setSearch}
// // // // // // //         showMenu={showMenu}
// // // // // // //         setShowMenu={setShowMenu}
// // // // // // //         />

// // // // // // //     <div className="home">
   
// // // // // // //        <Left />

// // // // // // //         <Middle 
// // // // // // //         handleSubmit={handleSubmit}
// // // // // // //         body ={body}
// // // // // // //         setBody ={setBody}
// // // // // // //         importFile ={importFile}
// // // // // // //         setImportFile ={setImportFile}
// // // // // // //         posts={posts}
// // // // // // //         setPosts={setPosts}
// // // // // // //         search={search}
// // // // // // //         setFriendsProfile={setFriendsProfile}
// // // // // // //         images={images}
// // // // // // //         setImages={setImages}

// // // // // // //         />

// // // // // // //         {/* <Right
// // // // // // //         showMenu={showMenu}
// // // // // // //         setShowMenu={setShowMenu}
// // // // // // //         following={following}
// // // // // // //         setFollowing={setFollowing}
// // // // // // //         /> */}
// // // // // // //     </div>

// // // // // // //     </div>
// // // // // // //   )
// // // // // // // }

// // // // // // // export default Home


// // // // // // // frontend/src/pages/Home/Home.js

// // // // // // import React, { useEffect, useState } from 'react';
// // // // // // import axios from 'axios';
// // // // // // import "../Home/Home.css"

// // // // // // import Left from "../../Components/LeftSide/Left"
// // // // // // import Middle from "../../Components/MiddleSide/Middle"
// // // // // // import Right from '../../Components/RightSide/Right'
// // // // // // import Nav from '../../Components/Navigation/Nav'
// // // // // // import moment from 'moment/moment'
// // // // // // import Post from '../../Components/Home/Post'
// // // // // // import { useNavigate } from 'react-router-dom';

// // // // // // const Home = ({ setFriendsProfile }) => {
// // // // // //   const [posts, setPosts] = useState([]); // Initialize as empty array
// // // // // //   const [search, setSearch] = useState('');
// // // // // //   const [showMenu, setShowMenu] = useState(false);
// // // // // //   const [likesCount, setLikesCount] = useState(0); // To track number of likes
// // // // // //   const navigate = useNavigate();

// // // // // //   // Authentication Check
// // // // // //   useEffect(() => {
// // // // // //     const token = localStorage.getItem('token');
// // // // // //     const tokenExpiry = localStorage.getItem('tokenExpiry');

// // // // // //     if (!token || !tokenExpiry) {
// // // // // //       // If token or expiry is missing, redirect to login
// // // // // //       navigate('/');
// // // // // //       return;
// // // // // //     }

// // // // // //     const currentTime = new Date().getTime();
// // // // // //     if (currentTime >= parseInt(tokenExpiry, 10)) {
// // // // // //       // If token has expired, clear storage and redirect to login
// // // // // //       localStorage.removeItem('token');
// // // // // //       localStorage.removeItem('tokenExpiry');
// // // // // //       navigate('/');
// // // // // //     }
// // // // // //   }, [navigate]);

// // // // // //   // Fetch images from backend on component mount
// // // // // //   useEffect(() => {
// // // // // //     fetchImages();
// // // // // //     // eslint-disable-next-line react-hooks/exhaustive-deps
// // // // // //   }, []);

// // // // // //   const fetchImages = async () => {
// // // // // //     const token = localStorage.getItem('token');
// // // // // //     try {
// // // // // //       const response = await axios.get('http://localhost:5000/get-images', {
// // // // // //         headers: {
// // // // // //           'Authorization': `Bearer ${token}`
// // // // // //         }
// // // // // //       });
// // // // // //       if (response.data.images) {
// // // // // //         setPosts(response.data.images);
// // // // // //       }
// // // // // //     } catch (error) {
// // // // // //       console.error('Error fetching images:', error.response?.data || error.message);
// // // // // //     }
// // // // // //   };

// // // // // //   const handleLike = async (imageIndex) => {
// // // // // //     const token = localStorage.getItem('token');
// // // // // //     try {
// // // // // //       await axios.post('http://localhost:5000/like-image', { image_index: imageIndex }, {
// // // // // //         headers: {
// // // // // //           'Authorization': `Bearer ${token}`
// // // // // //         }
// // // // // //       });
// // // // // //       setLikesCount(likesCount + 1);
// // // // // //       // Optionally, remove the liked post from the current list
// // // // // //       setPosts(posts.filter(post => post.index !== imageIndex));

// // // // // //       // After k likes, fetch recommendations
// // // // // //       const k = 5; // Define k as needed
// // // // // //       if ((likesCount + 1) % k === 0) {
// // // // // //         fetchRecommendations();
// // // // // //       }
// // // // // //     } catch (error) {
// // // // // //       console.error('Error liking image:', error.response?.data || error.message);
// // // // // //     }
// // // // // //   };

// // // // // //   const fetchRecommendations = async () => {
// // // // // //     const token = localStorage.getItem('token');
// // // // // //     try {
// // // // // //       const response = await axios.get('http://localhost:5000/recommend-images', {
// // // // // //         headers: {
// // // // // //           'Authorization': `Bearer ${token}`
// // // // // //         }
// // // // // //       });
// // // // // //       if (response.data.recommendations) {
// // // // // //         setPosts(response.data.recommendations);
// // // // // //       }
// // // // // //     } catch (error) {
// // // // // //       console.error('Error fetching recommendations:', error.response?.data || error.message);
// // // // // //     }
// // // // // //   };

// // // // // //   return (
// // // // // //     <div className='interface'>
// // // // // //       <Nav 
// // // // // //         search={search}
// // // // // //         setSearch={setSearch}
// // // // // //         showMenu={showMenu}
// // // // // //         setShowMenu={setShowMenu}
// // // // // //       />

// // // // // //       <div className="home">
// // // // // //         <Left />

// // // // // //         <div className="middle">
// // // // // //           {posts.length > 0 ? (
// // // // // //             posts.map(post => (
// // // // // //               <Post 
// // // // // //                 key={post.index}
// // // // // //                 artist={post.artist}
// // // // // //                 imageIndex={post.index}
// // // // // //                 description={`${post.genre}, ${post.style}`}
// // // // // //                 image={post.image}
// // // // // //                 onLike={() => handleLike(post.index)}
// // // // // //                 setFriendsProfile={setFriendsProfile}
// // // // // //               />
// // // // // //             ))
// // // // // //           ) : (
// // // // // //             <p>No posts available.</p>
// // // // // //           )}
// // // // // //         </div>

// // // // // //         {/* <Right
// // // // // //           showMenu={showMenu}
// // // // // //           setShowMenu={setShowMenu}
// // // // // //           following={following}
// // // // // //           setFollowing={setFollowing}
// // // // // //         /> */}
// // // // // //         {/* <Right /> */}
// // // // // //       </div>
// // // // // //     </div>
// // // // // //   );
// // // // // // }

// // // // // // export default Home;


// // // // // // frontend/src/pages/Home/Home.js

// // // // // import React, { useEffect, useState } from 'react';
// // // // // import axios from 'axios';
// // // // // import "../Home/Home.css";

// // // // // import Left from "../../Components/LeftSide/Left";
// // // // // import Middle from "../../Components/MiddleSide/Middle";
// // // // // import Right from '../../Components/RightSide/Right';
// // // // // import Nav from '../../Components/Navigation/Nav';
// // // // // import { useNavigate } from 'react-router-dom';
// // // // // import InfiniteScroll from 'react-infinite-scroll-component';
// // // // // import Post from '../../Components/Home/Post'; // Adjust the path if necessary

// // // // // const Home = ({ setFriendsProfile }) => {
// // // // //   const [posts, setPosts] = useState([]); // All fetched posts
// // // // //   const [displayedPosts, setDisplayedPosts] = useState([]); // Posts currently displayed
// // // // //   const [hasMore, setHasMore] = useState(true); // Indicator for more posts
// // // // //   const [search, setSearch] = useState('');
// // // // //   const [showMenu, setShowMenu] = useState(false);
// // // // //   const [likesCount, setLikesCount] = useState(0); // To track number of likes
// // // // //   const navigate = useNavigate();

// // // // //   const POSTS_PER_PAGE = 10; // Number of posts to load each time

// // // // //   // Authentication Check
// // // // //   useEffect(() => {
// // // // //     const token = localStorage.getItem('token');
// // // // //     const tokenExpiry = localStorage.getItem('tokenExpiry');

// // // // //     if (!token || !tokenExpiry) {
// // // // //       // If token or expiry is missing, redirect to login
// // // // //       navigate('/');
// // // // //       return;
// // // // //     }

// // // // //     const currentTime = new Date().getTime();
// // // // //     if (currentTime >= parseInt(tokenExpiry, 10)) {
// // // // //       // If token has expired, clear storage and redirect to login
// // // // //       localStorage.removeItem('token');
// // // // //       localStorage.removeItem('tokenExpiry');
// // // // //       navigate('/');
// // // // //     }
// // // // //   }, [navigate]);

// // // // //   // Fetch images from backend on component mount
// // // // //   useEffect(() => {
// // // // //     fetchImages();
// // // // //     // eslint-disable-next-line react-hooks/exhaustive-deps
// // // // //   }, []);

// // // // //   const fetchImages = async () => {
// // // // //     const token = localStorage.getItem('token');
// // // // //     try {
// // // // //       const response = await axios.get('http://localhost:5000/get-images', {
// // // // //         headers: {
// // // // //           'Authorization': `Bearer ${token}`
// // // // //         }
// // // // //       });
// // // // //       if (response.data.images) {
// // // // //         setPosts(response.data.images);
// // // // //         setDisplayedPosts(response.data.images.slice(0, POSTS_PER_PAGE));
// // // // //         if (response.data.images.length <= POSTS_PER_PAGE) {
// // // // //           setHasMore(false);
// // // // //         }
// // // // //       }
// // // // //     } catch (error) {
// // // // //       console.error('Error fetching images:', error.response?.data || error.message);
// // // // //     }
// // // // //   };

// // // // //   const fetchMorePosts = () => {
// // // // //     if (displayedPosts.length >= posts.length) {
// // // // //       setHasMore(false);
// // // // //       return;
// // // // //     }

// // // // //     // Simulate a delay for loading (optional)
// // // // //     setTimeout(() => {
// // // // //       const nextPosts = posts.slice(displayedPosts.length, displayedPosts.length + POSTS_PER_PAGE);
// // // // //       setDisplayedPosts([...displayedPosts, ...nextPosts]);
// // // // //       if (displayedPosts.length + nextPosts.length >= posts.length) {
// // // // //         setHasMore(false);
// // // // //       }
// // // // //     }, 1000);
// // // // //   };

// // // // //   const handleLike = async (imageIndex) => {
// // // // //     const token = localStorage.getItem('token');
// // // // //     try {
// // // // //       await axios.post('http://localhost:5000/like-image', { image_index: imageIndex }, {
// // // // //         headers: {
// // // // //           'Authorization': `Bearer ${token}`
// // // // //         }
// // // // //       });
// // // // //       setLikesCount(likesCount + 1);
// // // // //       // Optionally, remove the liked post from the current list
// // // // //       // setDisplayedPosts(displayedPosts.filter(post => post.index !== imageIndex));
// // // // //       // setPosts(posts.filter(post => post.index !== imageIndex));

// // // // //       // After k likes, fetch recommendations
// // // // //       const k = 5; // Define k as needed
// // // // //       if ((likesCount + 1) % k === 0) {
// // // // //         fetchRecommendations();
// // // // //       }
// // // // //     } catch (error) {
// // // // //       console.error('Error liking image:', error.response?.data || error.message);
// // // // //     }
// // // // //   };

// // // // //   const fetchRecommendations = async () => {
// // // // //     const token = localStorage.getItem('token');
// // // // //     try {
// // // // //       const response = await axios.get('http://localhost:5000/recommend-images', {
// // // // //         headers: {
// // // // //           'Authorization': `Bearer ${token}`
// // // // //         }
// // // // //       });
// // // // //       if (response.data.recommendations) {
// // // // //         const newPosts = response.data.recommendations;
// // // // //         setPosts([...posts, ...newPosts]);
// // // // //         setDisplayedPosts([...displayedPosts, ...newPosts.slice(0, POSTS_PER_PAGE)]);
// // // // //         if (newPosts.length <= POSTS_PER_PAGE) {
// // // // //           setHasMore(false);
// // // // //         }
// // // // //       }
// // // // //     } catch (error) {
// // // // //       console.error('Error fetching recommendations:', error.response?.data || error.message);
// // // // //     }
// // // // //   };

// // // // //   return (
// // // // //     <div className='interface'>
// // // // //       <Nav 
// // // // //         search={search}
// // // // //         setSearch={setSearch}
// // // // //         showMenu={showMenu}
// // // // //         setShowMenu={setShowMenu}
// // // // //       />

// // // // //       <div className="home">
// // // // //         <Left />

// // // // //         <div className="middle">
// // // // //           <InfiniteScroll
// // // // //             dataLength={displayedPosts.length}
// // // // //             next={fetchMorePosts}
// // // // //             hasMore={hasMore}
// // // // //             loader={<h4>Loading...</h4>}
// // // // //             endMessage={
// // // // //               <p style={{ textAlign: 'center' }}>
// // // // //                 <b>No more posts to display</b>
// // // // //               </p>
// // // // //             }
// // // // //           >
// // // // //             <div className="posts-grid">
// // // // //               {displayedPosts.map(post => (
// // // // //                 <Post 
// // // // //                   key={post.index}
// // // // //                   artist={post.artist}
// // // // //                   imageIndex={post.index}
// // // // //                   description={`${post.genre}, ${post.style}`}
// // // // //                   image={post.image}
// // // // //                   onLike={() => handleLike(post.index)}
// // // // //                   setFriendsProfile={setFriendsProfile}
// // // // //                   timestamp={post.timestamp} // Ensure this field exists
// // // // //                 />
// // // // //               ))}
// // // // //             </div>
// // // // //           </InfiniteScroll>
// // // // //         </div>

// // // // //         {/* <Right /> */}
// // // // //       </div>
// // // // //     </div>
// // // // //   );
// // // // // }

// // // // // export default Home;


// // // // // frontend/src/pages/Home/Home.js

// // // // import React, { useEffect, useState } from 'react';
// // // // import axios from 'axios';
// // // // import "../Home/Home.css";

// // // // import Left from "../../Components/LeftSide/Left";
// // // // import Nav from '../../Components/Navigation/Nav';
// // // // import { useNavigate } from 'react-router-dom';
// // // // import InfiniteScroll from 'react-infinite-scroll-component';
// // // // import Post from '../../Components/Home/Post'; // Adjust the path if necessary

// // // // const Home = ({ setFriendsProfile }) => {
// // // //   const [posts, setPosts] = useState([]); // All currently loaded posts
// // // //   const [hasMore, setHasMore] = useState(true); // Indicator for more posts
// // // //   const [search, setSearch] = useState('');
// // // //   const [showMenu, setShowMenu] = useState(false);
// // // //   const [likesCount, setLikesCount] = useState(0); // To track number of likes
// // // //   const [page, setPage] = useState(1); // Current page number
// // // //   const navigate = useNavigate();

// // // //   const POSTS_PER_PAGE = 5; // Number of posts to load each time

// // // //   // Authentication Check
// // // //   useEffect(() => {
// // // //     const token = localStorage.getItem('token');
// // // //     const tokenExpiry = localStorage.getItem('tokenExpiry');

// // // //     if (!token || !tokenExpiry) {
// // // //       // If token or expiry is missing, redirect to login
// // // //       navigate('/');
// // // //       return;
// // // //     }

// // // //     const currentTime = new Date().getTime();
// // // //     if (currentTime >= parseInt(tokenExpiry, 10)) {
// // // //       // If token has expired, clear storage and redirect to login
// // // //       localStorage.removeItem('token');
// // // //       localStorage.removeItem('tokenExpiry');
// // // //       navigate('/');
// // // //     }
// // // //   }, [navigate]);

// // // //   // Fetch initial posts on component mount
// // // //   useEffect(() => {
// // // //     fetchImages(page);
// // // //     // eslint-disable-next-line react-hooks/exhaustive-deps
// // // //   }, []);

// // // //   const fetchImages = async (currentPage) => {
// // // //     const token = localStorage.getItem('token');
// // // //     try {
// // // //       const response = await axios.get('http://localhost:5000/get-images', {
// // // //         headers: {
// // // //           'Authorization': `Bearer ${token}`
// // // //         },
// // // //         params: {
// // // //           page: currentPage,
// // // //           limit: POSTS_PER_PAGE
// // // //         }
// // // //       });

// // // //       if (response.data.images && response.data.images.length > 0) {
// // // //         setPosts(prevPosts => [...prevPosts, ...response.data.images]);

// // // //         // If fewer images than POSTS_PER_PAGE are returned, no more posts are available
// // // //         if (response.data.images.length < POSTS_PER_PAGE) {
// // // //           setHasMore(false);
// // // //         }
// // // //       } else {
// // // //         // If no images are returned, no more posts are available
// // // //         setHasMore(false);
// // // //       }
// // // //     } catch (error) {
// // // //       console.error('Error fetching images:', error.response?.data || error.message);
// // // //       // Optionally, set hasMore to false to prevent further fetch attempts
// // // //       setHasMore(false);
// // // //     }
// // // //   };

// // // //   const fetchMorePosts = () => {
// // // //     const nextPage = page + 1;
// // // //     fetchImages(nextPage);
// // // //     setPage(nextPage);
// // // //   };

// // // //   const handleLike = async (imageIndex) => {
// // // //     const token = localStorage.getItem('token');
// // // //     try {
// // // //       await axios.post('http://localhost:5000/like-image', { image_index: imageIndex }, {
// // // //         headers: {
// // // //           'Authorization': `Bearer ${token}`
// // // //         }
// // // //       });
// // // //       setLikesCount(prevCount => prevCount + 1);

// // // //       // After k likes, fetch recommendations
// // // //       const k = 5; // Define k as needed
// // // //       if ((likesCount + 1) % k === 0) {
// // // //         fetchRecommendations();
// // // //       }
// // // //     } catch (error) {
// // // //       console.error('Error liking image:', error.response?.data || error.message);
// // // //     }
// // // //   };

// // // //   const fetchRecommendations = async () => {
// // // //     const token = localStorage.getItem('token');
// // // //     try {
// // // //       const response = await axios.get('http://localhost:5000/recommend-images', {
// // // //         headers: {
// // // //           'Authorization': `Bearer ${token}`
// // // //         }
// // // //       });
// // // //       if (response.data.recommendations && response.data.recommendations.length > 0) {
// // // //         setPosts(prevPosts => [...prevPosts, ...response.data.recommendations]);

// // // //         // If fewer recommendations than POSTS_PER_PAGE are returned, no more posts are available
// // // //         if (response.data.recommendations.length < POSTS_PER_PAGE) {
// // // //           setHasMore(false);
// // // //         }
// // // //       } else {
// // // //         // If no recommendations are returned, you might choose to set hasMore to false
// // // //         // depending on your application's logic
// // // //         // setHasMore(false);
// // // //       }
// // // //     } catch (error) {
// // // //       console.error('Error fetching recommendations:', error.response?.data || error.message);
// // // //     }
// // // //   };

// // // //   return (
// // // //     <div className='interface'>
// // // //       <Nav 
// // // //         search={search}
// // // //         setSearch={setSearch}
// // // //         showMenu={showMenu}
// // // //         setShowMenu={setShowMenu}
// // // //       />

// // // //       <div className="home">
// // // //         <Left />

// // // //         <div className="middle">
// // // //           <InfiniteScroll
// // // //             dataLength={posts.length}
// // // //             next={fetchMorePosts}
// // // //             hasMore={hasMore}
// // // //             loader={<h4>Loading...</h4>}
// // // //             endMessage={
// // // //               <p style={{ textAlign: 'center' }}>
// // // //                 <b>No more posts to display</b>
// // // //               </p>
// // // //             }
// // // //           >
// // // //             <div className="posts-grid">
// // // //               {posts.map(post => (
// // // //                 <Post 
// // // //                   key={post.index}
// // // //                   artist={post.artist}
// // // //                   imageIndex={post.index}
// // // //                   description={`${post.genre}, ${post.style}`}
// // // //                   image={post.image}
// // // //                   onLike={() => handleLike(post.index)}
// // // //                   setFriendsProfile={setFriendsProfile}
// // // //                   timestamp={post.timestamp} // Ensure this field exists
// // // //                 />
// // // //               ))}
// // // //             </div>
// // // //           </InfiniteScroll>
// // // //         </div>

// // // //         {/* <Right /> */}
// // // //       </div>
// // // //     </div>
// // // //   );
// // // // }

// // // // export default Home;


// // // // frontend/src/pages/Home/Home.js

// // // import React, { useEffect, useState } from 'react';
// // // import axios from 'axios';
// // // import "../Home/Home.css";

// // // import Left from "../../Components/LeftSide/Left";
// // // import Nav from '../../Components/Navigation/Nav';
// // // import { useNavigate } from 'react-router-dom';
// // // import InfiniteScroll from 'react-infinite-scroll-component';
// // // import Post from '../../Components/Home/Post'; // Adjust the path if necessary

// // // const Home = ({ setFriendsProfile }) => {
// // //   const [posts, setPosts] = useState([]); // All currently loaded posts
// // //   const [hasMore, setHasMore] = useState(true); // Indicator for more posts
// // //   const [search, setSearch] = useState('');
// // //   const [showMenu, setShowMenu] = useState(false);
// // //   const [likesCount, setLikesCount] = useState(0); // To track number of likes
// // //   const navigate = useNavigate();

// // //   // Authentication Check
// // //   useEffect(() => {
// // //     const token = localStorage.getItem('token');
// // //     const tokenExpiry = localStorage.getItem('tokenExpiry');

// // //     if (!token || !tokenExpiry) {
// // //       // If token or expiry is missing, redirect to login
// // //       navigate('/');
// // //       return;
// // //     }

// // //     const currentTime = new Date().getTime();
// // //     if (currentTime >= parseInt(tokenExpiry, 10)) {
// // //       // If token has expired, clear storage and redirect to login
// // //       localStorage.removeItem('token');
// // //       localStorage.removeItem('tokenExpiry');
// // //       navigate('/');
// // //     }
// // //   }, [navigate]);

// // //   // Fetch initial recommendations on component mount
// // //   useEffect(() => {
// // //     fetchRecommendations();
// // //     // eslint-disable-next-line react-hooks/exhaustive-deps
// // //   }, []);

// // //   const fetchRecommendations = async () => {
// // //     const token = localStorage.getItem('token');
// // //     try {
// // //       const response = await axios.get('http://localhost:5000/recommend-images', {
// // //         headers: {
// // //           'Authorization': `Bearer ${token}`
// // //         }
// // //       });

// // //       if (response.data.recommendations && response.data.recommendations.length > 0) {
// // //         setPosts(prevPosts => [...prevPosts, ...response.data.recommendations]);

// // //         // If fewer images than expected are returned, no more posts are available
// // //         if (response.data.recommendations.length < 10) { // Assuming batch size is 10
// // //           setHasMore(false);
// // //         }
// // //       } else {
// // //         // If no images are returned, no more posts are available
// // //         setHasMore(false);
// // //       }
// // //     } catch (error) {
// // //       console.error('Error fetching recommendations:', error.response?.data || error.message);
// // //       // Optionally, set hasMore to false to prevent further fetch attempts
// // //       setHasMore(false);
// // //     }
// // //   };

// // //   const fetchMoreRecommendations = () => {
// // //     fetchRecommendations();
// // //   };

// // //   const handleLike = async (imageIndex) => {
// // //     const token = localStorage.getItem('token');
// // //     try {
// // //       await axios.post('http://localhost:5000/like-image', { image_index: imageIndex }, {
// // //         headers: {
// // //           'Authorization': `Bearer ${token}`
// // //         }
// // //       });
// // //       setLikesCount(prevCount => prevCount + 1);

// // //       // Fetch new recommendations immediately after a like
// // //       fetchRecommendations();
// // //     } catch (error) {
// // //       console.error('Error liking image:', error.response?.data || error.message);
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

// // //         <div className="middle">
// // //           <InfiniteScroll
// // //             dataLength={posts.length}
// // //             next={fetchMoreRecommendations}
// // //             hasMore={hasMore}
// // //             loader={<h4>Loading...</h4>}
// // //             endMessage={
// // //               <p style={{ textAlign: 'center' }}>
// // //                 <b>No more posts to display</b>
// // //               </p>
// // //             }
// // //           >
// // //             <div className="posts-grid">
// // //               {posts.map(post => (
// // //                 <Post 
// // //                   key={post.index}
// // //                   artist={post.artist}
// // //                   imageIndex={post.index}
// // //                   description={`${post.genre}, ${post.style}`}
// // //                   image={post.image}
// // //                   onLike={() => handleLike(post.index)}
// // //                   setFriendsProfile={setFriendsProfile}
// // //                   timestamp={post.timestamp} // Ensure this field exists
// // //                 />
// // //               ))}
// // //             </div>
// // //           </InfiniteScroll>
// // //         </div>

// // //         {/* <Right /> */}
// // //       </div>
// // //     </div>
// // //   );
// // // }

// // // export default Home;


// // // frontend/src/pages/Home/Home.js

// // import React, { useEffect, useState } from 'react';
// // import axios from 'axios';
// // import "../Home/Home.css";

// // import Left from "../../Components/LeftSide/Left";
// // import Nav from '../../Components/Navigation/Nav';
// // import { useNavigate } from 'react-router-dom';
// // import InfiniteScroll from 'react-infinite-scroll-component';
// // import Post from '../../Components/Home/Post'; // Adjust the path if necessary

// // const Home = ({ setFriendsProfile }) => {
// //   const [posts, setPosts] = useState([]); // All currently loaded posts
// //   const [hasMore, setHasMore] = useState(true); // Indicator for more posts
// //   const [search, setSearch] = useState('');
// //   const [showMenu, setShowMenu] = useState(false);
// //   // Removed likesCount as embeddings are updated immediately
// //   const navigate = useNavigate();

// //   // Authentication Check
// //   useEffect(() => {
// //     const token = localStorage.getItem('token');
// //     const tokenExpiry = localStorage.getItem('tokenExpiry');

// //     if (!token || !tokenExpiry) {
// //       // If token or expiry is missing, redirect to login
// //       navigate('/');
// //       return;
// //     }

// //     const currentTime = new Date().getTime();
// //     if (currentTime >= parseInt(tokenExpiry, 10)) {
// //       // If token has expired, clear storage and redirect to login
// //       localStorage.removeItem('token');
// //       localStorage.removeItem('tokenExpiry');
// //       navigate('/');
// //     }
// //   }, [navigate]);

// //   // Fetch initial recommendations on component mount
// //   useEffect(() => {
// //     fetchRecommendations();
// //     // eslint-disable-next-line react-hooks/exhaustive-deps
// //   }, []);

// //   const fetchRecommendations = async () => {
// //     const token = localStorage.getItem('token');
// //     try {
// //       const response = await axios.get('http://localhost:5000/recommend-images', {
// //         headers: {
// //           'Authorization': `Bearer ${token}`
// //         }
// //       });

// //       if (response.data.images && response.data.images.length > 0) {
// //         setPosts(prevPosts => [...prevPosts, ...response.data.images]);

// //         // If fewer images than expected are returned, no more posts are available
// //         if (response.data.images.length < 10) { // Batch size is 10
// //           setHasMore(false);
// //         }
// //       } else {
// //         // If no images are returned, no more posts are available
// //         setHasMore(false);
// //       }
// //     } catch (error) {
// //       console.error('Error fetching recommendations:', error.response?.data || error.message);
// //       // Optionally, set hasMore to false to prevent further fetch attempts
// //       setHasMore(false);
// //     }
// //   };

// //   const fetchMoreRecommendations = () => {
// //     fetchRecommendations();
// //   };

// //   const handleLike = async (imageIndex) => {
// //     const token = localStorage.getItem('token');
// //     try {
// //       await axios.post('http://localhost:5000/like-image', { image_index: imageIndex }, {
// //         headers: {
// //           'Authorization': `Bearer ${token}`
// //         }
// //       });

// //       // Fetch new recommendations immediately after a like
// //       fetchRecommendations();
// //     } catch (error) {
// //       console.error('Error liking image:', error.response?.data || error.message);
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

// //         <div className="middle">
// //           <InfiniteScroll
// //             dataLength={posts.length}
// //             next={fetchMoreRecommendations}
// //             hasMore={hasMore}
// //             loader={<h4>Loading...</h4>}
// //             endMessage={
// //               <p style={{ textAlign: 'center' }}>
// //                 <b>No more posts to display</b>
// //               </p>
// //             }
// //           >
// //             <div className="posts-grid">
// //               {posts.map(post => (
// //                 <Post 
// //                   key={`${post.index}-${post.timestamp}`} // Ensure uniqueness
// //                   artist={post.artist}
// //                   imageIndex={post.index}
// //                   description={`${post.genre}, ${post.style}`}
// //                   image={post.image}
// //                   onLike={() => handleLike(post.index)}
// //                   setFriendsProfile={setFriendsProfile}
// //                   timestamp={post.timestamp} // Ensure this field exists
// //                 />
// //               ))}
// //             </div>
// //           </InfiniteScroll>
// //         </div>

// //         {/* <Right /> */}
// //       </div>
// //     </div>
// //   );
// // }

// // export default Home;


// // frontend/src/pages/Home/Home.js

// import React, { useEffect, useState } from 'react';
// import axios from 'axios';
// import "../Home/Home.css";

// import Left from "../../Components/LeftSide/Left";
// import Nav from '../../Components/Navigation/Nav';
// import { useNavigate } from 'react-router-dom';
// import Post from '../../Components/Home/Post'; // Adjust the path if necessary

// const Home = ({ setFriendsProfile }) => {
//   const [posts, setPosts] = useState([]); // All currently loaded posts
//   const [hasMore, setHasMore] = useState(true); // Indicator for more posts
//   const [search, setSearch] = useState('');
//   const [showMenu, setShowMenu] = useState(false);
//   const [loadingMore, setLoadingMore] = useState(false); // Loading state for "Load More"
//   const navigate = useNavigate();

//   // Authentication Check
//   useEffect(() => {
//     const token = localStorage.getItem('token');
//     const tokenExpiry = localStorage.getItem('tokenExpiry');

//     if (!token || !tokenExpiry) {
//       // If token or expiry is missing, redirect to login
//       navigate('/');
//       return;
//     }

//     const currentTime = new Date().getTime();
//     if (currentTime >= parseInt(tokenExpiry, 10)) {
//       // If token has expired, clear storage and redirect to login
//       localStorage.removeItem('token');
//       localStorage.removeItem('tokenExpiry');
//       navigate('/');
//     }
//   }, [navigate]);

//   // Fetch initial recommendations on component mount
//   useEffect(() => {
//     fetchRecommendations();
//     // eslint-disable-next-line react-hooks/exhaustive-deps
//   }, []);

//   const fetchRecommendations = async () => {
//     const token = localStorage.getItem('token');
//     try {
//       setLoadingMore(true); // Set loading state
//       const response = await axios.get('http://localhost:5000/recommend-images', {
//         headers: {
//           'Authorization': `Bearer ${token}`
//         }
//       });

//       if (response.data.images && response.data.images.length > 0) {
//         setPosts(prevPosts => [...prevPosts, ...response.data.images]);

//         // If fewer images than expected are returned, no more posts are available
//         if (response.data.images.length < 10) { // Batch size is 10
//           setHasMore(false);
//         }
//       } else {
//         // If no images are returned, no more posts are available
//         setHasMore(false);
//       }
//     } catch (error) {
//       console.error('Error fetching recommendations:', error.response?.data || error.message);
//       setHasMore(false);
//     } finally {
//       setLoadingMore(false); // Reset loading state
//     }
//   };

//   const handleLike = async (imageIndex) => {
//     const token = localStorage.getItem('token');
//     try {
//       await axios.post('http://localhost:5000/like-image', { image_index: imageIndex }, {
//         headers: {
//           'Authorization': `Bearer ${token}`
//         }
//       });

//       // No need to fetch recommendations here, as the "Load More" button will handle it.
//     } catch (error) {
//       console.error('Error liking image:', error.response?.data || error.message);
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

//         <div className="middle">
//           <div className="posts-grid">
//             {posts.map(post => (
//               <Post 
//                 key={`${post.index}-${post.timestamp}`} // Ensure uniqueness
//                 artist={post.artist}
//                 imageIndex={post.index}
//                 description={`${post.genre}, ${post.style}`}
//                 image={post.image}
//                 onLike={() => handleLike(post.index)}
//                 setFriendsProfile={setFriendsProfile}
//                 timestamp={post.timestamp} // Ensure this field exists
//               />
//             ))}
//           </div>

//           {hasMore && (
//             <div className="load-more-container">
//               <button 
//                 onClick={fetchRecommendations} 
//                 className="load-more-button" 
//                 disabled={loadingMore}
//               >
//                 {loadingMore ? 'Loading...' : 'Load More'}
//               </button>
//             </div>
//           )}

//           {!hasMore && (
//             <p style={{ textAlign: 'center' }}>
//               <b>No more posts to display</b>
//             </p>
//           )}
//         </div>

//         {/* <Right /> */}
//       </div>
//     </div>
//   );
// }

// export default Home;


import React, { useEffect, useState } from 'react';
import axios from 'axios';
import "../Home/Home.css";

// import Censor from "./censor.jpg"
import Left from "../../Components/LeftSide/Left";
import Nav from '../../Components/Navigation/Nav';
import { useNavigate } from 'react-router-dom';
import Post from '../../Components/Home/Post'; // Adjust the path if necessary

const Home = ({ setFriendsProfile }) => {
  const [posts, setPosts] = useState([]); // All currently loaded posts
  const [hasMore, setHasMore] = useState(true); // Indicator for more posts
  const [search, setSearch] = useState('');
  const [showMenu, setShowMenu] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false); // Loading state for "Load More"
  const [collaborativeFiltering, setCollaborativeFiltering] = useState(false); // Toggle for collaborative filtering
  const navigate = useNavigate();

  // Authentication Check
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

  // Fetch initial recommendations on component mount
  useEffect(() => {
    fetchRecommendations();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [collaborativeFiltering]); // Refetch when collaborative filtering changes

  const fetchRecommendations = async () => {
    const token = localStorage.getItem('token');
    try {
      setLoadingMore(true);
      const endpoint = collaborativeFiltering 
        ? 'http://localhost:5000/recommend-images?collaborative_filtering=true' 
        : 'http://localhost:5000/recommend-images?collaborative_filtering=false';

      const response = await axios.get(endpoint, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.data.images && response.data.images.length > 0) {
        setPosts(prevPosts => [...prevPosts, ...response.data.images]);

        if (response.data.images.length < 10) { // Batch size is 10
          setHasMore(false);
        }
      } else {
        setHasMore(false);
      }
    } catch (error) {
      console.error('Error fetching recommendations:', error.response?.data || error.message);
      setHasMore(false);
    } finally {
      setLoadingMore(false);
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
    } catch (error) {
      console.error('Error liking image:', error.response?.data || error.message);
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

        <div className="middle">
          {/* Collaborative Filtering Toggle */}
          <div className="toggle-container">
            <label className="toggle-label">
              Activate Collaborative Filtering?
              <input 
                type="checkbox" 
                className="toggle-checkbox" 
                checked={collaborativeFiltering} 
                onChange={() => {
                  setPosts([]); // Reset posts when toggling
                  setHasMore(true); // Reset load state
                  setCollaborativeFiltering(!collaborativeFiltering);
                }} 
              />
              <span className="toggle-slider"></span>
            </label>
          </div>

          {/* <div className="posts-grid">
            {posts.map(post => (
              <Post 
                key={`${post.index}-${post.timestamp}`} 
                artist={post.artist}
                imageIndex={post.index}
                description={`${post.genre}, ${post.style}`}
                image={post.image}
                onLike={() => handleLike(post.index)}
                setFriendsProfile={setFriendsProfile}
                timestamp={post.timestamp}
              />
            ))}
          </div> */}
          <div className="posts-grid">
          {posts.map(post => {
            const sanitizedDescription = 
              post.genre.toLowerCase().includes("nude") || post.genre.toLowerCase().includes("naked") ||
              post.style.toLowerCase().includes("nude") || post.style.toLowerCase().includes("naked")
                ? "Content not available"
                : `${post.genre}, ${post.style}`;
            
            return (
              <Post 
                key={`${post.index}-${post.timestamp}`} 
                artist={post.artist}
                imageIndex={post.index}
                description={sanitizedDescription}
                image={post.image}
                onLike={() => handleLike(post.index)}
                setFriendsProfile={setFriendsProfile}
                timestamp={post.timestamp}
                iscensored={sanitizedDescription=="Content not available"?true:false}
              />
            );
          })}
        </div>


          {hasMore && (
            <div className="load-more-container">
              <button 
                onClick={fetchRecommendations} 
                className="load-more-button" 
                disabled={loadingMore}
              >
                {loadingMore ? 'Loading...' : 'Load More'}
              </button>
            </div>
          )}

          {!hasMore && (
            <p style={{ textAlign: 'center' }}>
              <b>No more posts to display</b>
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

export default Home;
