import React, { useState } from 'react'
import Home from './Pages/Home/Home'
import Chatbot from "./Pages/Home/Chatbot"
import Saved from "./Pages/Home/Saved"
import Lists from "./Pages/Home/Lists"
import Trending from "./Pages/Home/Trending"
import Profile from './Pages/Profile/Profile'
import FriendsId from "./Pages/FriendsId/FriendsId"
import { Route, Routes } from 'react-router-dom'
import Notification from './Pages/Notification/Notification'
import Login from './Pages/RegisterPage/Login'
import SignUp from './Pages/RegisterPage/SignUp'
import ProtectedRoute from './Components/ProtectedRoute';

const App = () => {
  const [friendProfile,setFriendsProfile] =useState([]);
  const [name,setname] = useState("Anonymous");

  return (
    <div className='App'>
      <Routes>
        <Route path='/home' element={<Home setFriendsProfile={setFriendsProfile}/>} />
        
        <Route path='/profile' element={ <Profile name={name} /> } />

        <Route path='/friendsId' element={<FriendsId friendProfile={friendProfile} name={name} />} />
      
        <Route path='/notification' element={<Notification name={name} />} />

        <Route path='/' element={<Login name={name} setname={setname}/>} />

        <Route path='/signup' element={<SignUp name={name}/>} />

        <Route path='/chatbot' element={<Chatbot setFriendsProfile={setFriendsProfile}/> } />
        <Route path='/saved' element={<Saved setFriendsProfile={setFriendsProfile}/> } />
        <Route path='/lists' element={<Lists setFriendsProfile={setFriendsProfile}/> } />
        <Route path='/trending' element={<Trending setFriendsProfile={setFriendsProfile}/> } />

        
      </Routes>
    </div>
  )
}

export default App
