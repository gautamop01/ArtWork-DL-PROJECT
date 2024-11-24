import React, { useState } from 'react'
import "../LeftSide/Left.css"
import {AiOutlineHome} from "react-icons/ai"
import {AiOutlineSearch} from "react-icons/ai"
import {FiTrendingUp} from "react-icons/fi"
import { Link } from 'react-router-dom';
import {BsBookmark} from "react-icons/bs"
import {RiFileListLine} from "react-icons/ri"
import {FiSettings} from "react-icons/fi"
import MoreHorizIcon from '@mui/icons-material/MoreHoriz';

import Profile from "../../assets/profile.jpg"


const Left = ({profileImg,
               modelDetails
              }) => {

  const [btnActive,setBtnActive] =useState("#")
  const [logOutExit,setLogOutExit] =useState(false)


  return (
    <div className="L-features">
      <Link to="/home" style={{textDecoration:"none",color:"black"}}>
        <div onClick={()=>setBtnActive("#")} id='L-box' className={btnActive === "#" ? "active" : ""} >
          <AiOutlineHome className='margin'/>
          <span>Explore</span>
        </div>
      </Link>
    
      {/* <div id='L-box' onClick={()=>setBtnActive("#explore")} className={btnActive === "#explore" ? "active" : ""}>
        <AiOutlineSearch
          className='margin'/>
         <span>Explore</span>
      </div> */}
      <Link to="/trending" style={{textDecoration:"none",color:"black"}}>
      <div id='L-box'  onClick={()=>setBtnActive("#trending")} className={btnActive === "#trending" ? "active" : ""}>
       <h1 className='notifi'>
          <FiTrendingUp 
           className='margin'/>
        </h1> 
        <span>Trending</span>
      </div>
      </Link>
      {/* <Link to="/lists" style={{textDecoration:"none",color:"black"}}>
      <div id='L-box' onClick={()=>setBtnActive("#lists")} className={btnActive === "#lists" ? "active" : ""}>
        <RiFileListLine
        className='margin'/>
        <span>Lists</span>
      </div>
      </Link> */}

      <Link to="/saved" style={{textDecoration:"none",color:"black"}}>
      <div id='L-box' onClick={()=>setBtnActive("#saved")} className={btnActive === "#saved" ? "active" : ""}>
        <BsBookmark
         className='margin'/>
        <span>Liked</span>
      </div>
      </Link>
      <Link to="/chatbot" id='chatbot' style={{marginTop:"8px",color:"black",textDecoration:'none'}}>
      <div id='L-box' onClick={()=>setBtnActive("#settings")} className={btnActive === "#settings" ? "active" : ""}>
        
        <FiSettings 
        className='margin'/>
        <span>Chatbot</span>
      </div>
      </Link>
      {/* <div id="L-box" onclick="uploadFile()"> */}
  {/* <span>Chatroom</span> */}
{/* </div> */}

      {/* <div className="left-user">
        <Link to="/profile" style={{textDecoration:"none",color:"black"}}>
          <div className="user-name-userid">
            <img src={profileImg ? (profileImg) : Profile} alt="" />
              <div className='L-user'>
                <h1>{modelDetails ? (modelDetails.ModelName) : "GauTam"}</h1>
                <span>{modelDetails ? (modelDetails.ModelUserName) : "@gautamop01"}</span>
            </div>
          </div>
        </Link>
        <MoreHorizIcon onClick={()=>setLogOutExit(!logOutExit)} className='vert'/>
          
          {logOutExit && (
            <div className="logOutExitContainer">
              <button>Add an existing account</button>
              <Link to="/" style={{width:"100%"}}><button>Log out @gautamop01</button></Link>
            </div>
          )}
      </div> */}

    </div>
  )
}

export default Left