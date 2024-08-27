import React from "react";

const Map: React.FC = () => {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        flexDirection: "column",
        height: "100vh",
      }}
    >
      <h2 style={{ color: "red", paddingTop: "30px", paddingBottom: "5px" }}>
        <strong>! Warning !</strong>
      </h2>
      <h3 style={{ color: "red", paddingTop: "14px", paddingBottom: "20px" }}>
        Epidemics : Malaria and dengue fever
      </h3>
      <h5
        style={{
          color: "red",
          paddingTop: "10px",
          paddingBottom: "10px",
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        A pandemic is an epidemic of infectious disease that spreads through
        human populations across a large region, multiple continents or
        globally.
        <br />
      </h5>
      <div
        style={{
          borderRadius: "10px", // Adjust the border radius value as needed
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: "600px",
            height: "400px",
            overflow: "hidden",
          }}
        >
          <iframe
            src="https://www.mapdevelopers.com/draw-circle-tool.php?circles=%5B%5B1000%2C31.3957107%2C75.5357675%2C%22%23AA143C%22%2C%22%23000000%22%2C0.4%5D%2C%5B1000%2C31.3957107%2C75.5357675%2C%22%23AA143C%22%2C%22%23FA143A%22%2C0.4%5D%2C%5B4000%2C31.3957107%2C75.5357675%2C%22%23AA143C%22%2C%22%23FA143A%22%2C0.4%5D%2C%5B3000%2C31.3957107%2C75.5357675%2C%22%23AA143C%22%2C%22%23FA143A%22%2C0.4%5D%2C%5B450.62%2C31.3967187%2C75.5392993%2C%22%2387AA14%22%2C%22%23F2FF40%22%2C0.4%5D%5D"
            width="600"
            height="700"
            style={{
              marginTop: "-270px",
              transform: "scale(1)",
            }}
            scrolling="no"
            allowFullScreen={true}
            loading="lazy"
            referrerPolicy="no-referrer-when-downgrade"
            title="Map"
          ></iframe>
        </div>
      </div>
    </div>
  );
};

export default Map;
