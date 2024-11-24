import { useState } from 'react'
import Left from '../../Components/LeftSide/Left'
import ProfileMiddle from '../../Components/Profile/ProfileMiddle'
import Right from '../../Components/RightSide/Right'
import Nav from '../../Components/Navigation/Nav'
import "../Profile/Profile.css"
import ProfileImg from "../../assets/profile.jpg"

const Profile = ({name}) => {

  const [following,setFollowing] =useState(3)
  const [search,setSearch] =useState("")

  const [showMenu,setShowMenu] =useState(false)

  const [images,setImages] =  useState(null)

  const [name1,setName]= useState("")
  const [userName,setUserName]= useState("")
  const [profileImg,setProfileImg] =useState(ProfileImg)

  const [modelDetails,setModelDetails] = useState(
    {
      ModelName:"GauTam",
      ModelUserName:"@gautamop01",
      ModelCountryName:"India",
      ModelJobName:"Web Developer"
    }
  )

  return (
    <div className='interface'>
        <Nav
        search={search}
        setSearch={setSearch}
        showMenu={showMenu}
        setShowMenu={setShowMenu}
        profileImg={profileImg}
        />
      <div className="home">
        <Left 
        following={following}
        setFollowing={setFollowing}
        profileImg={profileImg}
        modelDetails={modelDetails}
        
        />

        <ProfileMiddle 
        following={following}
        search={search}
        images={images}
        setImages={setImages}
        name={name}
        setName={setName}
        userName={userName}
        setUserName={setUserName}
        profileImg={profileImg}
        setProfileImg={setProfileImg}
        modelDetails={modelDetails}
        setModelDetails={setModelDetails}
        />
        
        {/* <Right 
        showMenu={showMenu}
        setShowMenu={setShowMenu}
        following={following}
        setFollowing={setFollowing}
        /> */}
      </div>
    </div>
  )
}

export default Profile