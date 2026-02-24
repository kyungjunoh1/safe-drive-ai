package com.safedriveai.backend.domain;

import jakarta.persistence.*;
import lombok.*;

import java.math.BigDecimal;
import java.time.LocalDateTime;

@Entity
@Table(name = "inspection_logs")
@Getter @Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class InspectionLog {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "driver_id", nullable = false)
    private Driver driver;

    @Column(nullable = false)
    private LocalDateTime inspectionTime;

    @Column(precision = 5, scale = 2)
    private BigDecimal faceScore;

    @Column(precision = 5, scale = 2)
    private BigDecimal voiceScore;

    @Column(precision = 5, scale = 2)
    private BigDecimal totalScore;

    @Column(nullable = false, length = 20)
    private String result;

    @Column(precision = 5, scale = 2)
    private BigDecimal alcoholProbability;

    @Column(length = 255)
    private String faceImagePath;

    @Column(columnDefinition = "TEXT")
    private String notes;

    @PrePersist
    protected void onCreate(){
        this.inspectionTime = LocalDateTime.now();
    }
}
