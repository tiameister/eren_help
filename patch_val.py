import sys

content = open('bluerov_led/validation.py').read()
old_part=content[content.find('def run_extract'):content.find('def validate_one')]
new_part = old_part[:old_part.find('dataset_folder =')] + '''dataset_folder = self.paths.dataset_folder(spec.dataset_name)

        import cv2
        from bluerov_led.pipeline import StreamingPipeline
        from bluerov_led.dataset_io import DatasetReader

        reader = DatasetReader(dataset_folder)
        frame_paths = reader.list_frame_paths()

        if not frame_paths:
            raise FileNotFoundError(
                f"No PNG frames found for dataset: {dataset_folder}"
            )

        print(f"  Running streaming extract on {spec.dataset_name} ...")
        
        # Instantiate the new online streaming pipeline
        streaming_pipeline = StreamingPipeline(config=self.config, distance_model_dict=self.distance_model.to_summary_dict())
        records = []
        
        for i, path in enumerate(frame_paths):
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            
            # Simulating real-time ingestion
            packet, candidates, mask_clean, record = streaming_pipeline.process_frame(frame, spec.dataset_name, i)
            record.file = path.name
            records.append(record)

        # Write output frame records using standard method
        from bluerov_led.dataset_io import ArtifactWriter
        ArtifactWriter.write_frame_records_csv(csv_path, records)
        return csv_path

    '''

with open('bluerov_led/validation.py', 'w') as f:
    f.write(content.replace(old_part, new_part))
print('Success')