import React from "react";
import styles from "../styles/SectonHeading.module.css";

interface Props {
  headingTitle?: string;
  color?: string;
}

function SectionHeading({
  headingTitle = "Placeholder",
  color = "black",
}: Props) {
  return (
    <div className={`center row`} style={{ borderColor: color, color }}>
      <div className={styles.headingContainer}>
        <h2 className={styles.heading}>{headingTitle}</h2>
      </div>
    </div>
  );
}

export default SectionHeading;
