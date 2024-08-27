import images from "./images";
const data = {
  services: [
    // {
    //   id: 1,
    //   picture: images.searchDoctor,
    //   title: "Skin Disease Detector",
    //   description:
    //     "A groundbreaking AI-driven tool for swiftly identifying and diagnosing various skin conditions, enhancing healthcare accessibility and efficiency.",
    //   url: "/",
    // },

    {
      id: 1,
      picture: images.consultation,
      title: "Heart Disease Predictor",
      description:
        "An advanced machine learning-based system that predicts heart disease risk, preventive healthcare with data-driven insights.",
      url: "/heart_disease",
    },
    {
      id: 2,
      picture: images.detailsInfo,
      title: "Diabetes Prediction Model",
      description:
        "A predictive model leveraging machine learning to forecast diabetes risk, empowering proactive health management through early detection.",
      url: "/dib_form",
    },
    {
      id: 3,
      picture: images.emergancyCare,
      title: "boT",
      description:
        "With its extensive training and knowledge, JivaBOT excels at providing accurate and insightful information on a wide range of medical topics.",
      url: "http://192.168.58.22:5000/",
    },
    {
      id: 5,
      picture: images.tracking,
      title: "Skin Disease Detection",
      description:
        "Skin Disease Detection employs advanced image analysis techniques to swiftly and accurately identify various dermatological conditions, facilitating prompt diagnosis and treatment.",
      url: "http://192.168.58.213:3000/",
    },
    {
      id: 5,
      picture: images.searchDoctor,
      title: "Location Based Epidemic Alert System",
      description:
        "Displaying epidemic warnings and visualizing affected areas on a map put on by verified doctors.",
      url: "/map",
    },
    {
      id: 5,
      picture: images.consultation,
      title: "JivaBLOGS",
      description:
        "Stay up to date with the latest health information by following blogs posted by verified doctors.",
      url: "http://192.168.58.22:3000/",
    },
  ],
};
export default data;
