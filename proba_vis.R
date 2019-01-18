#' Visualizations of the Proba-V data
#'

library("tidyverse")
library("reshape2")
library("EBImage")
theme_set(
  theme_bw() +
  theme(
    panel.grid =  element_blank(),
    panel.spacing = unit(0, "cm"),
    axis.text.y =  element_blank(),
    axis.ticks.y =  element_blank()
  )
)

## read in all the data
data_dir <- "/Users/krissankaran/Desktop/super-res/superres_data/train/NIR/"
id <- "imgset1157"

lr_files <- list.files(file.path(data_dir, id), "LR*", full.names = TRUE)
hr_file <- list.files(file.path(data_dir, id), "HR*", full.names = TRUE)
qm_files <- list.files(file.path(data_dir, id), "QM*", full.names = TRUE)
sm_file <- list.files(file.path(data_dir, id), "SM*", full.names = TRUE)
pred_file <- file.path("~/Desktop/train_preds", paste0(id, "_pred.png"))

lr <- lapply(lr_files, readImage)
qm <- lapply(qm_files, readImage)
hr <- readImage(hr_file)
sm <- readImage(sm_file)
pred <- readImage(pred_file)
R <- 4 # number of rows to plot

## reshape all the data into data frames
qm_df <- melt(lapply(qm, function(x) x[1:R, ])) %>%
  rename(quality = value) %>%
  mutate(quality = as.factor(quality))

lr_df <- melt(lapply(lr, function(x) x[1:R, ])) %>%
  left_join(qm_df) %>%
  mutate(
    Var2 = seq(1, nrow(hr), length.out = nrow(lr[[1]]))[Var2]
  ) %>%
  rename(view = L1)

sm_df <- melt(sm[1:R, ]) %>%
  rename(quality = value) %>%
  mutate(quality = as.factor(quality))

hr_df <- melt(hr[seq(1, 3 * R, 3), ]) %>%
  left_join(sm_df)

pred_df <- melt(pred[seq(1, 3 * R, 3), ]) %>%
  left_join(sm_df)

## generate the plot
ggplot() +
  geom_point(
    data = lr_df,
    aes(
      x = Var2,
      y = value,
      color = quality
    ),
    alpha = 0.8,
    size = 0.1
  ) +
  geom_point(
    data = hr_df,
    aes(
      x = Var2,
      y = value,
      col = quality
    ),
    size = 0.5,
    shape = 2
  ) +
  geom_point(
    data = pred_df,
    aes(
      x = Var2,
      y = value,
      col = quality
    ),
    size = 0.5,
    shape = 3
  ) +
  scale_x_continuous(expand = c(0, 0)) +
  xlim(0, 40) +
  guides(colour = guide_legend(override.aes = list(alpha = 1, size = 1))) +
  scale_color_manual(values = c("#cd5b45", "#5d3954")) +
  facet_grid(Var1 ~ .)

#ggsave(sprintf("notes/figure/views_comparison_%s.png", id), width = 4, height = 7, dpi = 400)
