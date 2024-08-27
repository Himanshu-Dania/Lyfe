"use client";
import Services from "@/components/Services";
import data from "@/constants/data";
import style from "@/styles/Map.module.css";
function Home() {
  return (
    <div>
      <Services services={data.services} />
      <div className={style.mapContainer}>{/* <MapDisease /> */}</div>
    </div>
  );
}
export default Home;
