import React from 'react';
import backgroundImage from '../../../public/blogs-desktop.jpg';

const LandingPage = () => {
  const backgroundStyle = {
    backgroundImage: `url("./blogs-desktop.jpg")`,
    backgroundSize: 'cover',
    backgroundPosition: 'center',
    minHeight: '70vh', // Ensure the image covers the entire viewport height
    display: 'flex',
    alignItems: 'center',
    // justifyContent: 'center',
     textAlign: 'center',
    color: 'black', // Text color changed to black
    fontFamily: 'sans-serif', // You can adjust the font family
  };

  const titleStyle = {
    color: '#113551',
    fontSize: '3.125rem',
    fontWeight: 700,
    lineHeight: 1.2,
     paddingBottom: '5', // Adjust padding value as needed
     textAlign:'left',
     paddingLeft :'15'
  };
  
  const pstyle = {
    color: '#1c1b1b',
    fontSize: '1.5625rem',
    fontWeight: 200,
    lineHeight: 1.4,
    paddingLeft : '150'
    
  };
  

  return (
    <div style={backgroundStyle}>
     <div style={{ paddingLeft: '150px',paddingBottom :'30px' }}> {/* Added paddingLeft style */}
  <h1 style={titleStyle} className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold mb-4">
    Jiva Blogs
  </h1>
  <p style={pstyle} className="text-lg sm:text-xl md:text-2xl">
    Personal Stories. Expert Perspectives.
  </p>
</div>
    </div>
  );
};

export default LandingPage;
