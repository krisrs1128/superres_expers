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
data_dir <- "/Users/krissankaran/Downloads/max_errors/"

lr_files <- list.files(data_dir, "low*", full.names = TRUE)
hr_file <- list.files(data_dir, "HR_gt*", full.names = TRUE)
pred_file <- list.files(data_dir, "HR_pred*", full.names = TRUE)

lr <- lapply(lr_files, readImage)
hr <- readImage(hr_file)
pred <- readImage(pred_file)
R0 <- 100
R <- 5 # number of rows to plot

## reshape all the data into data frames
lr_df <- melt(lapply(lr, function(x) x[R0:(R0 + R), ])) %>%
  mutate(
    Var2 = seq(1, nrow(hr), length.out = nrow(lr[[1]]))[Var2],
    type = "low_res"
  ) %>%
  rename(view = L1)

pred_df <- melt(pred[seq(3 * R0, 3 * (R0 +  R), 3), ]) %>%
  mutate(type = "pred")
hr_df <- melt(hr[seq(3 * R0, 3 * (R0 + R), 3), ]) %>%
  mutate(type = "truth")

df <- lr_df %>%
  full_join(pred_df) %>%
  full_join(hr_df)

## generate the plot
ggplot() +
  geom_point(
    data = df,
    aes(
      x = Var2,
      y = value,
      col = type
    ),
    size = 0.1
  ) +
  scale_x_continuous(expand = c(0, 0)) +
  guides(colour = guide_legend(override.aes = list(alpha = 1, size = 1))) +
  ## scale_color_manual(values = c("#cd5b45", "#5d3954")) +
  facet_grid(Var1 ~ .)

#ggsave(sprintf("notes/figure/views_comparison_%s.png", id), width = 4, height = 7, dpi = 400)

image(hr - pred)
hist(hr - pred)
plot(as.vector(hr), as.vector(pred), col = rgb(0, 0, 0, 0.1))
abline(b = 1, a = 0, col = "red")
