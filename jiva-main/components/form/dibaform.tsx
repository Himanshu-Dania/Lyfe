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
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  SelectValue,
  SelectTrigger,
  SelectContent,
  SelectItem,
  Select,
} from "@/components/ui/select";
import Formf from "../Formf";

const formSchema = z.object({
  Age: z.number().min(0).max(99), // Restricting to a single digit
  Gender_Female: z.number().min(0).max(1), // Assuming only 0 or 1 are valid
  Gender_Male: z.number().min(0).max(1), // Assuming only 0 or 1 are valid
  Polyuria_No: z.number().min(0).max(1), // Assuming only 0 or 1 are valid
  Polyuria_Yes: z.number().min(0).max(1), // Assuming only 0 or 1 are valid
  Polydipsia_No: z.number().min(0).max(1), // Assuming only 0 or 1 are valid
  Polydipsia_Yes: z.number().min(0).max(1), // Assuming only 0 or 1 are valid
  suddenweightloss_No: z.number().min(0).max(1), // Assuming only 0 or 1 are valid
  suddenweightloss_Yes: z.number().min(0).max(1), // Assuming only 0 or 1 are valid
  weakness_No: z.number().min(0).max(1), // Assuming only 0 or 1 are valid
  weakness_Yes: z.number().min(0).max(1), // Assuming only 0 or 1 are valid
  Polyphagia_No: z.number().min(0).max(1), // Assuming only 0 or 1 are valid
  Polyphagia_Yes: z.number().min(0).max(1), // Assuming only 0 or 1 are valid
  Genitalthrush_No: z.number().min(0).max(1), // Assuming only 0 or 1 are valid
  Genitalthrush_Yes: z.number().min(0).max(1), // Assuming only 0 or 1 are valid
  visualblurring_No: z.number().min(0).max(1), // Assuming only 0 or 1 are valid
  visualblurring_Yes: z.number().min(0).max(1), // Assuming only 0 or 1 are valid
  Itching_No: z.number().min(0).max(1), // Assuming only 0 or 1 are valid
  Itching_Yes: z.number().min(0).max(1), // Assuming only 0 or 1 are valid
  Irritability_No: z.number().min(0).max(1), // Assuming only 0 or 1 are valid
  Irritability_Yes: z.number().min(0).max(1), // Assuming only 0 or 1 are valid
  delayedhealing_No: z.number().min(0).max(1), // Assuming only 0 or 1 are valid
  delayedhealing_Yes: z.number().min(0).max(1), // Assuming only 0 or 1 are valid
  partialparesis_No: z.number().min(0).max(1), // Assuming only 0 or 1 are valid
  partialparesis_Yes: z.number().min(0).max(1), // Assuming only 0 or 1 are valid
  musclestiffness_No: z.number().min(0).max(1), // Assuming only 0 or 1 are valid
  musclestiffness_Yes: z.number().min(0).max(1), // Assuming only 0 or 1 are valid
  Alopecia_No: z.number().min(0).max(1), // Assuming only 0 or 1 are valid
  Alopecia_Yes: z.number().min(0).max(1), // Assuming only 0 or 1 are valid
  Obesity_No: z.number().min(0).max(1), // Assuming only 0 or 1 are valid
  Obesity_Yes: z.number().min(0).max(1), // Assuming only 0 or 1 are valid // Assuming only 0 or 1 are valid
});

export default function DibForm() {
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    // defaultValues: {
    //   Age: 0,
    //   Gender_Female: 1,
    //   Gender_Male: 0,
    //   Polyuria_No: 1,
    //   Polyuria_Yes: 0,
    //   Polydipsia_No: 1,
    //   Polydipsia_Yes: 0,
    //   suddenweightloss_No: 1,
    //   suddenweightloss_Yes: 0,
    //   weakness_No: 1,
    //   weakness_Yes: 0,
    //   Polyphagia_No: 1,
    //   Polyphagia_Yes: 0,
    //   Genitalthrush_No: 1,
    //   Genitalthrush_Yes: 0,
    //   visualblurring_No: 1,
    //   visualblurring_Yes: 0,
    //   Itching_No: 1,
    //   Itching_Yes: 0,
    //   Irritability_No: 1,
    //   Irritability_Yes: 0,
    //   delayedhealing_No: 1,
    //   delayedhealing_Yes: 0,
    //   partialparesis_No: 1,
    //   partialparesis_Yes: 0,
    //   musclestiffness_No: 1,
    //   musclestiffness_Yes: 0,
    //   Alopecia_No: 1,
    //   Alopecia_Yes: 0,
    //   Obesity_No: 1,
    //   Obesity_Yes: 0,
    // },
  });
  const [predictData, setpredictData] = useState<any>(null);

  const HandleSubmit = (values: z.infer<typeof formSchema>) => {
    const formData = {
      Age: values.Age,
      Gender_Female: values.Gender_Female,
      Gender_Male: values.Gender_Male,
      Polyuria_No: values.Polyphagia_No,
      Polyuria_Yes: values.Polydipsia_Yes,
      Polydipsia_No: values.Polydipsia_No,
      Polydipsia_Yes: values.Polydipsia_Yes,
      suddenweightloss_No: values.suddenweightloss_No,
      suddenweightloss_Yes: values.suddenweightloss_Yes,
      weakness_No: values.weakness_No,
      weakness_Yes: values.weakness_Yes,
      Polyphagia_No: values.Polydipsia_No,
      Polyphagia_Yes: values.Polydipsia_Yes,
      Genitalthrush_No: values.Genitalthrush_No,
      Genitalthrush_Yes: values.Genitalthrush_Yes,
      visualblurring_No: values.visualblurring_No,
      visualblurring_Yes: values.visualblurring_Yes,
      Itching_No: values.Itching_No,
      Itching_Yes: values.Itching_Yes,
      Irritability_No: values.Irritability_No,
      Irritability_Yes: values.Irritability_Yes,
      delayedhealing_No: values.delayedhealing_No,
      delayedhealing_Yes: values.delayedhealing_Yes,
      partialparesis_No: values.partialparesis_No,
      partialparesis_Yes: values.partialparesis_Yes,
      musclestiffness_No: values.musclestiffness_No,
      musclestiffness_Yes: values.musclestiffness_Yes,
      Alopecia_No: values.musclestiffness_Yes,
      Alopecia_Yes: values.Alopecia_Yes,
      Obesity_No: values.Obesity_No,
      Obesity_Yes: values.Obesity_Yes,
    };
    console.log(formData);
    fetch("http://127.0.0.1:7000/predict_api", {
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
        const predictionProbability = data[0]?.prediction_probability; // Use optional chaining to handle potential undefined values
        const prediction = data[1]?.prediction;
        console.log(predictionProbability);
        // Update state to display the response data
        setpredictData({
          predictionProbability: predictionProbability,
          prediction: prediction,
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
              ? "There are chances for heart condition"
              : "The chances are slim (though do consult a doctor in case of any numb feeling or prelonged thirst.)"}
          </p>
          <p className="mb-1">
            Prediction Probability:{" "}
            {predictData.prediction === 1
              ? (predictData.predictionProbability[1] * 100).toFixed(2)
              : (predictData.predictionProbability[0] * 100).toFixed(2)}{" "}
            %
          </p>
        </div>
      )}

      <Form {...form}>
        <form
          onSubmit={form.handleSubmit(HandleSubmit)}
          className=" w-full stretch gap-10 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4"
        >
          <FormField
            control={form.control}
            name="Age"
            render={({ field }) => {
              return (
                <FormItem>
                  <FormLabel>Age</FormLabel>
                  <FormControl>
                    <Input
                      placeholder="Input Age"
                      type="number"
                      {...field}
                      min="0"
                      max="99"
                      onChange={(e) =>
                        form.setValue("Age", parseInt(e.target.value))
                      }
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              );
            }}
          />
          <FormField
            control={form.control}
            name="Gender_Male"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Gender</FormLabel>
                <div className="flex gap-4">
                  <Button
                    type="button"
                    onClick={() => {
                      form.setValue("Gender_Male", 1);
                      form.setValue("Gender_Female", 0);
                    }}
                    className={`${
                      form.getValues("Gender_Male") === 1
                        ? "bg-blue-500"
                        : "bg-gray-300"
                    }`}
                  >
                    Male
                  </Button>
                  <Button
                    type="button"
                    onClick={() => {
                      form.setValue("Gender_Male", 0);
                      form.setValue("Gender_Female", 1);
                    }}
                    className={`${
                      form.getValues("Gender_Female") === 1
                        ? "bg-blue-500"
                        : "bg-gray-300"
                    }`}
                  >
                    Female
                  </Button>
                </div>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="Alopecia_Yes"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Alopecia</FormLabel>
                <div className="flex gap-4">
                  <Button
                    type="button"
                    onClick={() => {
                      form.setValue("Alopecia_Yes", 1);
                      form.setValue("Alopecia_No", 0);
                    }}
                    className={`${
                      form.getValues("Alopecia_Yes") === 1
                        ? "bg-blue-500"
                        : "bg-gray-300"
                    }`}
                  >
                    Yes
                  </Button>
                  <Button
                    type="button"
                    onClick={() => {
                      form.setValue("Alopecia_Yes", 0);
                      form.setValue("Alopecia_No", 1);
                    }}
                    className={`${
                      form.getValues("Alopecia_No") === 1
                        ? "bg-blue-500"
                        : "bg-gray-300"
                    }`}
                  >
                    No
                  </Button>
                </div>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="Polyuria_Yes"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Polyuria</FormLabel>
                <div className="flex gap-4">
                  <Button
                    type="button"
                    onClick={() => {
                      form.setValue("Polyuria_Yes", 1);
                      form.setValue("Polyuria_No", 0);
                    }}
                    className={`${
                      form.getValues("Polyuria_Yes") === 1
                        ? "bg-blue-500"
                        : "bg-gray-300"
                    }`}
                  >
                    Yes
                  </Button>
                  <Button
                    type="button"
                    onClick={() => {
                      form.setValue("Polyuria_Yes", 0);
                      form.setValue("Polyuria_No", 1);
                    }}
                    className={`${
                      form.getValues("Polyuria_No") === 1
                        ? "bg-blue-500"
                        : "bg-gray-300"
                    }`}
                  >
                    No
                  </Button>
                </div>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="Polydipsia_Yes"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Polydipisa</FormLabel>
                <div className="flex gap-4">
                  <Button
                    type="button"
                    onClick={() => {
                      form.setValue("Polydipsia_Yes", 1);
                      form.setValue("Polydipsia_No", 0);
                    }}
                    className={`${
                      form.getValues("Polydipsia_Yes") === 1
                        ? "bg-blue-500"
                        : "bg-gray-300"
                    }`}
                  >
                    Yes
                  </Button>
                  <Button
                    type="button"
                    onClick={() => {
                      form.setValue("Polydipsia_Yes", 0);
                      form.setValue("Polydipsia_No", 1);
                    }}
                    className={`${
                      form.getValues("Polydipsia_No") === 1
                        ? "bg-blue-500"
                        : "bg-gray-300"
                    }`}
                  >
                    No
                  </Button>
                </div>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="suddenweightloss_Yes"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Sudden Weight Loss?</FormLabel>
                <div className="flex gap-4">
                  <Button
                    type="button"
                    onClick={() => {
                      form.setValue("suddenweightloss_Yes", 1);
                      form.setValue("suddenweightloss_No", 0);
                    }}
                    className={`${
                      form.getValues("suddenweightloss_Yes") === 1
                        ? "bg-blue-500"
                        : "bg-gray-300"
                    }`}
                  >
                    Yes
                  </Button>
                  <Button
                    type="button"
                    onClick={() => {
                      form.setValue("suddenweightloss_Yes", 0);
                      form.setValue("suddenweightloss_No", 1);
                    }}
                    className={`${
                      form.getValues("suddenweightloss_No") === 1
                        ? "bg-blue-500"
                        : "bg-gray-300"
                    }`}
                  >
                    No
                  </Button>
                </div>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="weakness_Yes"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Weakness?</FormLabel>
                <div className="flex gap-4">
                  <Button
                    type="button"
                    onClick={() => {
                      form.setValue("weakness_Yes", 1);
                      form.setValue("weakness_No", 0);
                    }}
                    className={`${
                      form.getValues("weakness_Yes") === 1
                        ? "bg-blue-500"
                        : "bg-gray-300"
                    }`}
                  >
                    Yes
                  </Button>
                  <Button
                    type="button"
                    onClick={() => {
                      form.setValue("weakness_Yes", 0);
                      form.setValue("weakness_No", 1);
                    }}
                    className={`${
                      form.getValues("weakness_No") === 1
                        ? "bg-blue-500"
                        : "bg-gray-300"
                    }`}
                  >
                    No
                  </Button>
                </div>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="Polyphagia_Yes"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Polyphagia</FormLabel>
                <div className="flex gap-4">
                  <Button
                    type="button"
                    onClick={() => {
                      form.setValue("Polyphagia_Yes", 1);
                      form.setValue("Polyphagia_No", 0);
                    }}
                    className={`${
                      form.getValues("Polyphagia_Yes") === 1
                        ? "bg-blue-500"
                        : "bg-gray-300"
                    }`}
                  >
                    Yes
                  </Button>
                  <Button
                    type="button"
                    onClick={() => {
                      form.setValue("Polyphagia_Yes", 0);
                      form.setValue("Polyphagia_No", 1);
                    }}
                    className={`${
                      form.getValues("Polyphagia_No") === 1
                        ? "bg-blue-500"
                        : "bg-gray-300"
                    }`}
                  >
                    No
                  </Button>
                </div>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="Genitalthrush_Yes"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Genitial thrust?</FormLabel>
                <div className="flex gap-4">
                  <Button
                    type="button"
                    onClick={() => {
                      form.setValue("Genitalthrush_Yes", 1);
                      form.setValue("Genitalthrush_No", 0);
                    }}
                    className={`${
                      form.getValues("Genitalthrush_Yes") === 1
                        ? "bg-blue-500"
                        : "bg-gray-300"
                    }`}
                  >
                    Yes
                  </Button>
                  <Button
                    type="button"
                    onClick={() => {
                      form.setValue("Genitalthrush_Yes", 0);
                      form.setValue("Genitalthrush_No", 1);
                    }}
                    className={`${
                      form.getValues("Genitalthrush_No") === 1
                        ? "bg-blue-500"
                        : "bg-gray-300"
                    }`}
                  >
                    No
                  </Button>
                </div>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="Itching_Yes"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Itching?</FormLabel>
                <div className="flex gap-4">
                  <Button
                    type="button"
                    onClick={() => {
                      form.setValue("Itching_Yes", 1);
                      form.setValue("Itching_No", 0);
                    }}
                    className={`${
                      form.getValues("Itching_Yes") === 1
                        ? "bg-blue-500"
                        : "bg-gray-300"
                    }`}
                  >
                    Yes
                  </Button>
                  <Button
                    type="button"
                    onClick={() => {
                      form.setValue("Itching_Yes", 0);
                      form.setValue("Itching_No", 1);
                    }}
                    className={`${
                      form.getValues("Itching_No") === 1
                        ? "bg-blue-500"
                        : "bg-gray-300"
                    }`}
                  >
                    No
                  </Button>
                </div>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="Irritability_Yes"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Irritability?</FormLabel>
                <div className="flex gap-4">
                  <Button
                    type="button"
                    onClick={() => {
                      form.setValue("Irritability_Yes", 1);
                      form.setValue("Irritability_No", 0);
                    }}
                    className={`${
                      form.getValues("Irritability_Yes") === 1
                        ? "bg-blue-500"
                        : "bg-gray-300"
                    }`}
                  >
                    Yes
                  </Button>
                  <Button
                    type="button"
                    onClick={() => {
                      form.setValue("Irritability_Yes", 0);
                      form.setValue("Irritability_No", 1);
                    }}
                    className={`${
                      form.getValues("Irritability_No") === 1
                        ? "bg-blue-500"
                        : "bg-gray-300"
                    }`}
                  >
                    No
                  </Button>
                </div>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="visualblurring_Yes"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Visual Bluring?</FormLabel>
                <div className="flex gap-4">
                  <Button
                    type="button"
                    onClick={() => {
                      form.setValue("visualblurring_Yes", 1);
                      form.setValue("visualblurring_No", 0);
                    }}
                    className={`${
                      form.getValues("visualblurring_Yes") === 1
                        ? "bg-blue-500"
                        : "bg-gray-300"
                    }`}
                  >
                    Yes
                  </Button>
                  <Button
                    type="button"
                    onClick={() => {
                      form.setValue("visualblurring_Yes", 0);
                      form.setValue("visualblurring_No", 1);
                    }}
                    className={`${
                      form.getValues("visualblurring_No") === 1
                        ? "bg-blue-500"
                        : "bg-gray-300"
                    }`}
                  >
                    No
                  </Button>
                </div>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="delayedhealing_Yes"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Delayed Healing?</FormLabel>
                <div className="flex gap-4">
                  <Button
                    type="button"
                    onClick={() => {
                      form.setValue("delayedhealing_Yes", 1);
                      form.setValue("delayedhealing_No", 0);
                    }}
                    className={`${
                      form.getValues("delayedhealing_Yes") === 1
                        ? "bg-blue-500"
                        : "bg-gray-300"
                    }`}
                  >
                    Yes
                  </Button>
                  <Button
                    type="button"
                    onClick={() => {
                      form.setValue("delayedhealing_Yes", 0);
                      form.setValue("delayedhealing_No", 1);
                    }}
                    className={`${
                      form.getValues("delayedhealing_No") === 1
                        ? "bg-blue-500"
                        : "bg-gray-300"
                    }`}
                  >
                    No
                  </Button>
                </div>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="partialparesis_Yes"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Partial Paresis?</FormLabel>
                <div className="flex gap-4">
                  <Button
                    type="button"
                    onClick={() => {
                      form.setValue("partialparesis_Yes", 1);
                      form.setValue("partialparesis_No", 0);
                    }}
                    className={`${
                      form.getValues("partialparesis_Yes") === 1
                        ? "bg-blue-500"
                        : "bg-gray-300"
                    }`}
                  >
                    Yes
                  </Button>
                  <Button
                    type="button"
                    onClick={() => {
                      form.setValue("partialparesis_Yes", 0);
                      form.setValue("partialparesis_No", 1);
                    }}
                    className={`${
                      form.getValues("partialparesis_No") === 1
                        ? "bg-blue-500"
                        : "bg-gray-300"
                    }`}
                  >
                    No
                  </Button>
                </div>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="musclestiffness_Yes"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Muscle Stiffness?</FormLabel>
                <div className="flex gap-4">
                  <Button
                    type="button"
                    onClick={() => {
                      form.setValue("musclestiffness_Yes", 1);
                      form.setValue("musclestiffness_No", 0);
                    }}
                    className={`${
                      form.getValues("musclestiffness_Yes") === 1
                        ? "bg-blue-500"
                        : "bg-gray-300"
                    }`}
                  >
                    Yes
                  </Button>
                  <Button
                    type="button"
                    onClick={() => {
                      form.setValue("musclestiffness_Yes", 0);
                      form.setValue("musclestiffness_No", 1);
                    }}
                    className={`${
                      form.getValues("musclestiffness_No") === 1
                        ? "bg-blue-500"
                        : "bg-gray-300"
                    }`}
                  >
                    No
                  </Button>
                </div>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="Obesity_Yes"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Obesity?</FormLabel>
                <div className="flex gap-4">
                  <Button
                    type="button"
                    onClick={() => {
                      form.setValue("Obesity_Yes", 1);
                      form.setValue("Obesity_No", 0);
                    }}
                    className={`${
                      form.getValues("Obesity_Yes") === 1
                        ? "bg-blue-500"
                        : "bg-gray-300"
                    }`}
                  >
                    Yes
                  </Button>
                  <Button
                    type="button"
                    onClick={() => {
                      form.setValue("Obesity_Yes", 0);
                      form.setValue("Obesity_No", 1);
                    }}
                    className={`${
                      form.getValues("Obesity_No") === 1
                        ? "bg-blue-500"
                        : "bg-gray-300"
                    }`}
                  >
                    No
                  </Button>
                </div>
                <FormMessage />
              </FormItem>
            )}
          />

          <Button type="submit" className="w-full">
            Submit
          </Button>
        </form>
      </Form>
    </main>
  );
}
