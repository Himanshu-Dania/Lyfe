"use client";
import React from "react";
import SectionHeading from "./SectionHeading";
import styles from "../styles/Services.module.css";
import ServiceCard from "./ServiceCard";
import { useSession } from "next-auth/react";
interface Service {
  id: number;
  picture: string;
  title: string;
  description: string;
  url: string;
}

interface Props {
  services: Service[];
}

function Services({ services }: Props) {
  const { data: session } = useSession();
  return (
    <section className={styles.container}>
      <div className="margin-on-side">
        <SectionHeading
          headingTitle={`Hello, ${session ? session.user.name : "User"}`}
        />
        <div className={`row ${styles.serviceList}`}>
          {services.map((item) => (
            <ServiceCard service={item} key={item.id} />
          ))}
        </div>
      </div>
    </section>
  );
}

export default Services;
