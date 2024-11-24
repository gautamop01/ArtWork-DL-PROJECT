import React from 'react'
import Feedposts from './Feedposts'
import "../Home/Homepage.css"


const Chatroom = ({posts,setPosts,setFriendsProfile,images}) => {
  return (
    <main className='Chatroom'>
        
        {posts.length ? <Feedposts 
                        images={images}
                        posts={posts}
                        setPosts={setPosts}
                        setFriendsProfile={setFriendsProfile}
                        /> 
        :
        (<p style={{textAlign:"center",marginTop:"40px"}}>
            NO POSTS ARE HERE
        </p>)
        }
    </main>
  )
}

export default Homepage