library(curatedMetagenomicData)
library(dplyr)
library(Matrix)
library(arrow)

# locate relavent studies by disease name
# for not only selected diseases change distinct filtering to >
relevant_studies <- sampleMetadata %>%
  group_by(study_name) %>%
  filter("CRC" %in% disease & "healthy" %in% disease, 
         n_distinct(disease) > 2) %>%
  pull(study_name) %>%
  unique()

cat(relevant_studies)

# get relevant files from each study 
for (study in relevant_studies) {
  
  # Filter metadata for the current study in the loop
  study_meta <- sampleMetadata %>%
    filter(study_name == study, disease %in% c("CRC", "healthy"))
  
  # Download the data for the current study
  tse <- returnSamples(study_meta, dataType = "relative_abundance")
  
  # Define the prefix so files are named correctly
  prefix <- paste0(study, "_")
  
  # Get metadata and add sample_id column
  meta_df <- as.data.frame(colData(tse))
  meta_df$sample_id <- rownames(meta_df)
  meta_df <- meta_df %>% relocate(sample_id)
  
  # Get matrix, transpose it (samples as rows), and add sample_id column
  abund_df <- as.data.frame(t(assay(tse, "relative_abundance")))
  abund_df$sample_id <- rownames(abund_df)
  abund_df <- abund_df %>% relocate(sample_id)
  
  # save metadata files
  write.csv(meta_df, file = paste0(prefix, "metadata.csv"), row.names = FALSE)

  # save bacteria info files
  write.csv(abund_df, file = paste0(prefix, "abundances.csv"), row.names = FALSE)

}

cat("download successful\n")
