'use client'

import Image from 'next/image'
import Link from 'next/link'
import React, { useEffect, useState } from 'react'
import classes from './blogCard.module.css'
import { useSession } from 'next-auth/react'
import { AiFillLike, AiOutlineLike } from 'react-icons/ai'
import bgImg from '../../../public/blogs-desktop.jpg'

const BlogCard = ({ blog: { title, desc, imageUrl, likes, authorId, _id } }) => {
  const { data: session } = useSession()
  const [isLiked, setIsLiked] = useState(false)
  const [blogLikes, setBlogLikes] = useState(0)


  useEffect(() => {
    session && likes && setIsLiked(likes.includes(session?.user?._id))
    session && likes && setBlogLikes(likes.length)
  }, [likes, session])

  const handleLike = async () => {
    try {
      const res = await fetch(`http://localhost:3000/api/blog/${_id}/like`, {
        headers: {
          'Authorization': `Bearer ${session?.user?.accessToken}`
        },
        method: 'PUT'
      })

      console.log(res)
      if (res.ok) {
        if (isLiked) {
          setIsLiked(prev => !prev)
          setBlogLikes(prev => prev - 1)
        } else {
          setIsLiked(prev => !prev)
          setBlogLikes(prev => prev + 1)
        }
      }
    } catch (error) {
      console.log(error)
    }
  }

  const [showMore, setShowMore] = useState(false);

  const truncatedDesc = desc.split(' ').slice(0, 50).join(' '); // Get first 100 words

  const handleToggleShowMore = () => {
    setShowMore(!showMore);
  };

  return (
   
       
    <div className={classes.maincontainer}>
    <div className={classes.container}>
      <div className={classes.wrapper}>
        <Link className={classes.imgContainer} href={`/blog/${_id}`}>
          <Image src={imageUrl} width="350" height="350" />
        </Link>
        <div className={classes.blogData}>
          <div className={classes.left}>
            <h3>{title}</h3>
            <p>
            {showMore ? desc : truncatedDesc} {/* Show truncated or full description */}
            {!showMore && desc.length > 100 && ( // If description is longer than 100 words, show "Read More" link
              <a className = {classes.readMore}href={`/blog/${_id}`}>Read More....</a>
            )}
          </p>

      
          
          </div>
          <div className={classes.right}>
            {blogLikes} {" "} {isLiked
              ? (<AiFillLike onClick={handleLike} size={20} />)
              : (<AiOutlineLike onClick={handleLike} size={20} />)}
          </div>
        </div>
      </div>
    </div>
    </div>
   
    
  )
}

export default BlogCard