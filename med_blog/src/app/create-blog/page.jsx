'use client'

import { useRouter } from 'next/navigation'
import React, { useState } from 'react'
import { AiOutlineFileImage } from 'react-icons/ai'
import { ToastContainer, toast } from 'react-toastify'
import 'react-toastify/dist/ReactToastify.css'
import { useSession } from 'next-auth/react'
import classes from './createBlog.module.css'

const CreateBlog = () => {
    const CLOUD_NAME = 'dzh3eecsg'
    const UPLOAD_PRESET = 'fyjcg2lg'

    const [title, setTitle] = useState('')
    const [desc, setDesc] = useState('')
    const [category, setCategory] = useState("Nature")
    const [photo, setPhoto] = useState('')

    const { data: session, status } = useSession()
    const router = useRouter()


    if (status === 'loading') {
        return <p>Loading...</p>
    }

    if (status === 'unauthenticated') {
        return <p className={classes.accessDenied}>
            Access Denied
        </p>
    }

    const handleSubmit = async (e) => {
        e.preventDefault()

        if (!photo || !title || !category || !desc) {
            toast.error("All fields are required")
            return
        }

        try {
            const imageUrl = await uploadImage()

            const res = await fetch(`http://localhost:3000/api/blog`, {
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${session?.user?.accessToken}`
                },
                method: 'POST',
                body: JSON.stringify({ title, desc, category, imageUrl, authorId: session?.user?._id })
            })

            if (!res.ok) {
                throw new Error("Error occured")
            }

            const blog = await res.json()

            router.push(`/blog/${blog?._id}`)
        } catch (error) {
            console.log(error)
        }
    }

    const uploadImage = async () => {
        if (!photo) return

        const formData = new FormData()

        formData.append("file", photo)
        formData.append("upload_preset", UPLOAD_PRESET)

        try {
            const res = await fetch(`https://api.cloudinary.com/v1_1/${CLOUD_NAME}/image/upload`, {
                method: "POST",
                body: formData
            })

            const data = await res.json()

            const imageUrl = data['secure_url']

            return imageUrl
        } catch (error) {
            console.log(error)
        }
    }

    return (
        <div className={classes.container}>
            <div className={classes.wrapper}>
                <h2>Create Post</h2>
                <form onSubmit={handleSubmit}>
                    <input type="text" placeholder='Title...' onChange={(e) => setTitle(e.target.value)} />
                    <textarea placeholder='Description...' onChange={(e) => setDesc(e.target.value)} />
                    <select value={category} onChange={(e) => setCategory(e.target.value)}>
                        <option value="Wellness and Lifestyle">Wellness and Lifestyle</option>
                        <option value="Health Tips and Advice">Health Tips and Advice</option>
                        <option value="Medical Conditions and Diseases">Medical Conditions and Diseases</option>
                        <option value="Patient Stories and Experiences">Patient Stories and Experiences</option>
                        <option value="Public Health and Advocacy">Public Health and Advocacy</option>
                    </select>
                    <label htmlFor='image'>
                        Upload Image <AiOutlineFileImage />
                    </label>
                    <input id='image' type="file" style={{ display: 'none' }} onChange={(e) => setPhoto(e.target.files[0])} />
                    <button className={classes.createBlog}>Create</button>
                </form>
            </div>
            <ToastContainer />
        </div>
    )
}

export default CreateBlog