import React from "react";
import Image from "next/image";
import Link from "next/link";
import styles from "../styles/ServiceCard.module.css";

interface Service {
  url: string;
  picture: string;
  title: string;
  description: string;
}

interface Props {
  service: Service;
}

function ServiceCard({ service }: Props) {
  return (
    <div className={styles.cardmainContainer}>
      <Link href={`${service.url}`} passHref>
        <div className={styles.cardContainer}>
          <div className={`${styles.imageContainer}`}>
            <div>
              <Image src={service.picture} alt="Health is Wealth" />
            </div>
          </div>
          <h3 className={`${styles.cardHeading}`}>{service.title}</h3>
          <p className={`${styles.cardDes} text-light-gray`}>
            {service.description}
          </p>
        </div>
      </Link>
    </div>
  );
}

export default ServiceCard;
