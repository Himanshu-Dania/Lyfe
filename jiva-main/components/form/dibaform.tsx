import * as z from "zod";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useState } from "react";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form"; // Adjust the import path as necessary
import { Button } from "@/components/ui/button"; // Adjust the import path as necessary

const formSchema = z.object({
  Age: z.number().min(0).max(99),
  Gender: z.boolean(), // Added Gender as a boolean
  Polyuria: z.boolean(),
  Polydipsia: z.boolean(),
  suddenweightloss: z.boolean(),
  weakness: z.boolean(),
  Polyphagia: z.boolean(),
  Genitalthrush: z.boolean(),
  visualblurring: z.boolean(),
  Itching: z.boolean(),
  Irritability: z.boolean(),
  delayedhealing: z.boolean(),
  partialparesis: z.boolean(),
  musclestiffness: z.boolean(),
  Alopecia: z.boolean(),
  Obesity: z.boolean(),
});

type FormFieldType = {
  label: string;
  name: keyof z.infer<typeof formSchema>;
  type: "boolean";
};

const formFields: FormFieldType[] = [
  { label: "Polyuria", name: "Polyuria", type: "boolean" },
  { label: "Polydipsia", name: "Polydipsia", type: "boolean" },
  { label: "Sudden Weight Loss", name: "suddenweightloss", type: "boolean" },
  { label: "Weakness", name: "weakness", type: "boolean" },
  { label: "Polyphagia", name: "Polyphagia", type: "boolean" },
  { label: "Genital Thrush", name: "Genitalthrush", type: "boolean" },
  { label: "Visual Blurring", name: "visualblurring", type: "boolean" },
  { label: "Itching", name: "Itching", type: "boolean" },
  { label: "Irritability", name: "Irritability", type: "boolean" },
  { label: "Delayed Healing", name: "delayedhealing", type: "boolean" },
  { label: "Partial Paresis", name: "partialparesis", type: "boolean" },
  { label: "Muscle Stiffness", name: "musclestiffness", type: "boolean" },
  { label: "Alopecia", name: "Alopecia", type: "boolean" },
  { label: "Obesity", name: "Obesity", type: "boolean" },
];

export default function DibForm() {
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
  });
  const [predictData, setPredictData] = useState<any>(null);

  const handleSubmit = (values: z.infer<typeof formSchema>) => {
    const formData = {
      Age: values.Age,
      Gender: values.Gender, // Include Gender in the form data
      Polyuria: values.Polyuria,
      Polydipsia: values.Polydipsia,
      suddenweightloss: values.suddenweightloss,
      weakness: values.weakness,
      Polyphagia: values.Polyphagia,
      Genitalthrush: values.Genitalthrush,
      visualblurring: values.visualblurring,
      Itching: values.Itching,
      Irritability: values.Irritability,
      delayedhealing: values.delayedhealing,
      partialparesis: values.partialparesis,
      musclestiffness: values.musclestiffness,
      Alopecia: values.Alopecia,
      Obesity: values.Obesity,
    };

    console.log("Form Data Submitted:", formData); // Log the form data

    fetch("https://diabetesdetection-g7bgascsdgbjavd4.westindia-01.azurewebsites.net/predict_api", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(formData),
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        return response.json();
      })
      .then((data) => {
        console.log("API Response Data:", data); // Log the response data
        const predictionProbability = data.prediction_probability[0];
        const prediction = data.prediction_probability[1];

        setPredictData({
          predictionProbability : predictionProbability,
          prediction : prediction,
        });
        console.log("Success:", data);
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  };

  return (
    <main className="flex flex-col w-full pt-16 items-center justify-center ">
      {predictData && (
        <div
          className={`px-4 py-8 mr-8 rounded-md shadow-md ${
            predictData.prediction === 1 ? "bg-red-100" : "bg-green-100"
          }`}
        >
          <h2 className="text-lg font-semibold mb-2">Results:</h2>
          <p className="mb-1">
            Prediction:{" "}
            {predictData.prediction === 1
              ? "There are chances for diabetes."
              : "The chances are slim (though do consult a doctor in case of any symptoms)."}
          </p>
          <p className="mb-1">
            Prediction Probability:{" "}
            {/* {Array.isArray(predictData.predictionProbability) && predictData.predictionProbability[0]
              ? (predictData.predictionProbability[0] * 100).toFixed(2)
              : "N/A"}% */}
            {predictData.prediction === 1
              ? (predictData.predictionProbability*100).toFixed(2)
              : (predictData.predictionProbability * 100).toFixed(2).toString()}{" "}
            %
          </p>
        </div>
      )}

      <Form {...form}>
        <form
          onSubmit={form.handleSubmit(handleSubmit)}
          className="w-full stretch gap-10 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4"
        >
          {/* Separate age input */}
          <FormField
            control={form.control}
            name="Age"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Age</FormLabel>
                <FormControl>
                  <input
                    placeholder="Input Age"
                    type="number"
                    {...field}
                    className="border rounded px-2 py-1"
                    onChange={(e) => field.onChange(parseInt(e.target.value) || 0)}
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />

          {/* Gender field */}
          <FormField
            control={form.control}
            name="Gender"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Gender</FormLabel>
                <div className="flex gap-4">
                  <Button
                    type="button"
                    onClick={() => field.onChange(true)}
                    className={field.value === true ? "bg-blue-500" : "bg-gray-300"}
                  >
                    Male
                  </Button>
                  <Button
                    type="button"
                    onClick={() => field.onChange(false)}
                    className={field.value === false ? "bg-blue-500" : "bg-gray-300"}
                  >
                    Female
                  </Button>
                </div>
                <FormMessage />
              </FormItem>
            )}
          />

          {/* Render boolean fields */}
          {formFields.map(({ label, name }) => (
            <FormField
              key={name}
              control={form.control}
              name={name}
              render={({ field }) => (
                <FormItem>
                  <FormLabel>{label}</FormLabel>
                  <div className="flex gap-4">
                    <Button
                      type="button"
                      onClick={() => field.onChange(true)}
                      className={field.value === true ? "bg-blue-500" : "bg-gray-300"}
                    >
                      Yes
                    </Button>
                    <Button
                      type="button"
                      onClick={() => field.onChange(false)}
                      className={field.value === false ? "bg-blue-500" : "bg-gray-300"}
                    >
                      No
                    </Button>
                  </div>
                  <FormMessage />
                </FormItem>
              )}
            />
          ))}
          <Button type="submit" className="col-span-full mt-4">
            Submit
          </Button>
        </form>
      </Form>
    </main>
  );
}
