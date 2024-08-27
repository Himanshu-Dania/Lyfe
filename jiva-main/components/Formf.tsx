import React from "react";
import { FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Button } from "@/components/ui/button";

interface FormfProps {
  control: any;
  name: string;
  label: string;
  options: { label: string; value: any }[];
  form: any;
}

const Formf: React.FC<FormfProps> = ({
  control,
  name,
  label,
  options,
  form,
}) => {
  return (
    <FormItem>
      <FormLabel>{label}</FormLabel>
      <div className="flex gap-4">
        {options.map((option, index) => (
          <Button
            key={index}
            type="button"
            onClick={() => {
              form.setValue(name, option.value);
            }}
            className={`${
              form.getValues(name) === option.value
                ? "bg-blue-500"
                : "bg-gray-300"
            }`}
          >
            {option.label}
          </Button>
        ))}
      </div>
      <FormMessage />
    </FormItem>
  );
};

export default Formf;
