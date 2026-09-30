; ModuleID = 'pipeline.bc'
source_filename = "pipeline.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@.str = private unnamed_addr constant [9 x i8] c"/bin/gen\00", align 1
@.str.1 = private unnamed_addr constant [8 x i8] c"/bin/up\00", align 1
@__px_handle = internal unnamed_addr global i32 -1, align 4
@.str.3 = private unnamed_addr constant [6 x i8] c"posix\00", align 1
@str = private unnamed_addr constant [12 x i8] c"pipeline ok\00", align 1

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = alloca [5 x i64], align 16
  %2 = alloca [5 x i64], align 16
  %3 = alloca [2 x i32], align 4
  %4 = alloca i32, align 4
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %3) #5
  %5 = load i32, ptr @__px_handle, align 4, !tbaa !5
  %6 = icmp slt i32 %5, 0
  br i1 %6, label %7, label %9

7:                                                ; preds = %0
  %8 = tail call i32 @__vm_cap_resolve(ptr noundef nonnull @.str.3, i64 noundef 5) #5
  store i32 %8, ptr @__px_handle, align 4, !tbaa !5
  br label %9

9:                                                ; preds = %0, %7
  %10 = phi i32 [ %8, %7 ], [ %5, %0 ]
  %11 = ptrtoint ptr %3 to i64
  %12 = call i64 @__vm_host_call(i32 noundef %10, i32 noundef 23, i64 noundef %11, i64 noundef 0, i64 noundef 0, i64 noundef 0) #5
  %13 = and i64 %12, 4294967295
  %14 = icmp eq i64 %13, 0
  br i1 %14, label %15, label %120

15:                                               ; preds = %9
  %16 = load i32, ptr %3, align 4, !tbaa !5
  %17 = getelementptr inbounds [2 x i32], ptr %3, i64 0, i64 1
  %18 = load i32, ptr %17, align 4, !tbaa !5
  %19 = load i32, ptr @__px_handle, align 4, !tbaa !5
  %20 = icmp slt i32 %19, 0
  br i1 %20, label %21, label %23

21:                                               ; preds = %15
  %22 = call i32 @__vm_cap_resolve(ptr noundef nonnull @.str.3, i64 noundef 5) #5
  store i32 %22, ptr @__px_handle, align 4, !tbaa !5
  br label %23

23:                                               ; preds = %15, %21
  %24 = phi i32 [ %22, %21 ], [ %19, %15 ]
  %25 = call i64 @__vm_host_call(i32 noundef %24, i32 noundef 25, i64 noundef 1, i64 noundef 0, i64 noundef 0, i64 noundef 0) #5
  %26 = and i64 %25, 2147483648
  %27 = icmp eq i64 %26, 0
  br i1 %27, label %28, label %120

28:                                               ; preds = %23
  %29 = load i32, ptr @__px_handle, align 4, !tbaa !5
  %30 = icmp slt i32 %29, 0
  br i1 %30, label %31, label %33

31:                                               ; preds = %28
  %32 = call i32 @__vm_cap_resolve(ptr noundef nonnull @.str.3, i64 noundef 5) #5
  store i32 %32, ptr @__px_handle, align 4, !tbaa !5
  br label %33

33:                                               ; preds = %28, %31
  %34 = phi i32 [ %32, %31 ], [ %29, %28 ]
  %35 = sext i32 %18 to i64
  %36 = call i64 @__vm_host_call(i32 noundef %34, i32 noundef 24, i64 noundef %35, i64 noundef 1, i64 noundef 0, i64 noundef 0) #5
  %37 = load i32, ptr @__px_handle, align 4, !tbaa !5
  %38 = icmp slt i32 %37, 0
  br i1 %38, label %39, label %41

39:                                               ; preds = %33
  %40 = call i32 @__vm_cap_resolve(ptr noundef nonnull @.str.3, i64 noundef 5) #5
  store i32 %40, ptr @__px_handle, align 4, !tbaa !5
  br label %41

41:                                               ; preds = %33, %39
  %42 = phi i32 [ %40, %39 ], [ %37, %33 ]
  %43 = call i64 @__vm_host_call(i32 noundef %42, i32 noundef 6, i64 noundef %35, i64 noundef 0, i64 noundef 0, i64 noundef 0) #5
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %4) #5
  store i32 0, ptr %4, align 4, !tbaa !5
  call void @llvm.lifetime.start.p0(i64 40, ptr nonnull %2) #5
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 16 dereferenceable(40) %2, i8 0, i64 40, i1 false)
  store i64 ptrtoint (ptr @.str to i64), ptr %2, align 16, !tbaa !9
  %44 = load i32, ptr @__px_handle, align 4, !tbaa !5
  %45 = icmp slt i32 %44, 0
  br i1 %45, label %46, label %48

46:                                               ; preds = %41
  %47 = call i32 @__vm_cap_resolve(ptr noundef nonnull @.str.3, i64 noundef 5) #5
  store i32 %47, ptr @__px_handle, align 4, !tbaa !5
  br label %48

48:                                               ; preds = %41, %46
  %49 = phi i32 [ %47, %46 ], [ %44, %41 ]
  %50 = ptrtoint ptr %2 to i64
  %51 = call i64 @__vm_host_call(i32 noundef %49, i32 noundef 62, i64 noundef %50, i64 noundef 0, i64 noundef 0, i64 noundef 0) #5
  call void @llvm.lifetime.end.p0(i64 40, ptr nonnull %2) #5
  %52 = icmp slt i64 %51, 0
  br i1 %52, label %118, label %53

53:                                               ; preds = %48
  %54 = load i32, ptr @__px_handle, align 4, !tbaa !5
  %55 = icmp slt i32 %54, 0
  br i1 %55, label %56, label %58

56:                                               ; preds = %53
  %57 = call i32 @__vm_cap_resolve(ptr noundef nonnull @.str.3, i64 noundef 5) #5
  store i32 %57, ptr @__px_handle, align 4, !tbaa !5
  br label %58

58:                                               ; preds = %53, %56
  %59 = phi i32 [ %57, %56 ], [ %54, %53 ]
  %60 = ptrtoint ptr %4 to i64
  %61 = call i64 @__vm_host_call(i32 noundef %59, i32 noundef 28, i64 noundef %51, i64 noundef %60, i64 noundef 0, i64 noundef 0) #5
  %62 = trunc i64 %61 to i32
  %63 = trunc i64 %51 to i32
  %64 = icmp eq i32 %62, %63
  br i1 %64, label %65, label %118

65:                                               ; preds = %58
  %66 = load i32, ptr @__px_handle, align 4, !tbaa !5
  %67 = icmp slt i32 %66, 0
  br i1 %67, label %68, label %70

68:                                               ; preds = %65
  %69 = call i32 @__vm_cap_resolve(ptr noundef nonnull @.str.3, i64 noundef 5) #5
  store i32 %69, ptr @__px_handle, align 4, !tbaa !5
  br label %70

70:                                               ; preds = %65, %68
  %71 = phi i32 [ %69, %68 ], [ %66, %65 ]
  %72 = and i64 %25, 2147483647
  %73 = call i64 @__vm_host_call(i32 noundef %71, i32 noundef 24, i64 noundef %72, i64 noundef 1, i64 noundef 0, i64 noundef 0) #5
  %74 = load i32, ptr @__px_handle, align 4, !tbaa !5
  %75 = icmp slt i32 %74, 0
  br i1 %75, label %76, label %78

76:                                               ; preds = %70
  %77 = call i32 @__vm_cap_resolve(ptr noundef nonnull @.str.3, i64 noundef 5) #5
  store i32 %77, ptr @__px_handle, align 4, !tbaa !5
  br label %78

78:                                               ; preds = %70, %76
  %79 = phi i32 [ %77, %76 ], [ %74, %70 ]
  %80 = call i64 @__vm_host_call(i32 noundef %79, i32 noundef 6, i64 noundef %72, i64 noundef 0, i64 noundef 0, i64 noundef 0) #5
  %81 = load i32, ptr @__px_handle, align 4, !tbaa !5
  %82 = icmp slt i32 %81, 0
  br i1 %82, label %83, label %85

83:                                               ; preds = %78
  %84 = call i32 @__vm_cap_resolve(ptr noundef nonnull @.str.3, i64 noundef 5) #5
  store i32 %84, ptr @__px_handle, align 4, !tbaa !5
  br label %85

85:                                               ; preds = %78, %83
  %86 = phi i32 [ %84, %83 ], [ %81, %78 ]
  %87 = sext i32 %16 to i64
  %88 = call i64 @__vm_host_call(i32 noundef %86, i32 noundef 24, i64 noundef %87, i64 noundef 0, i64 noundef 0, i64 noundef 0) #5
  %89 = load i32, ptr @__px_handle, align 4, !tbaa !5
  %90 = icmp slt i32 %89, 0
  br i1 %90, label %91, label %93

91:                                               ; preds = %85
  %92 = call i32 @__vm_cap_resolve(ptr noundef nonnull @.str.3, i64 noundef 5) #5
  store i32 %92, ptr @__px_handle, align 4, !tbaa !5
  br label %93

93:                                               ; preds = %85, %91
  %94 = phi i32 [ %92, %91 ], [ %89, %85 ]
  %95 = call i64 @__vm_host_call(i32 noundef %94, i32 noundef 6, i64 noundef %87, i64 noundef 0, i64 noundef 0, i64 noundef 0) #5
  call void @llvm.lifetime.start.p0(i64 40, ptr nonnull %1) #5
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 16 dereferenceable(40) %1, i8 0, i64 40, i1 false)
  store i64 ptrtoint (ptr @.str.1 to i64), ptr %1, align 16, !tbaa !9
  %96 = load i32, ptr @__px_handle, align 4, !tbaa !5
  %97 = icmp slt i32 %96, 0
  br i1 %97, label %98, label %100

98:                                               ; preds = %93
  %99 = call i32 @__vm_cap_resolve(ptr noundef nonnull @.str.3, i64 noundef 5) #5
  store i32 %99, ptr @__px_handle, align 4, !tbaa !5
  br label %100

100:                                              ; preds = %93, %98
  %101 = phi i32 [ %99, %98 ], [ %96, %93 ]
  %102 = ptrtoint ptr %1 to i64
  %103 = call i64 @__vm_host_call(i32 noundef %101, i32 noundef 62, i64 noundef %102, i64 noundef 0, i64 noundef 0, i64 noundef 0) #5
  call void @llvm.lifetime.end.p0(i64 40, ptr nonnull %1) #5
  %104 = icmp slt i64 %103, 0
  br i1 %104, label %118, label %105

105:                                              ; preds = %100
  %106 = load i32, ptr @__px_handle, align 4, !tbaa !5
  %107 = icmp slt i32 %106, 0
  br i1 %107, label %108, label %110

108:                                              ; preds = %105
  %109 = call i32 @__vm_cap_resolve(ptr noundef nonnull @.str.3, i64 noundef 5) #5
  store i32 %109, ptr @__px_handle, align 4, !tbaa !5
  br label %110

110:                                              ; preds = %105, %108
  %111 = phi i32 [ %109, %108 ], [ %106, %105 ]
  %112 = call i64 @__vm_host_call(i32 noundef %111, i32 noundef 28, i64 noundef %103, i64 noundef %60, i64 noundef 0, i64 noundef 0) #5
  %113 = trunc i64 %112 to i32
  %114 = trunc i64 %103 to i32
  %115 = icmp eq i32 %113, %114
  br i1 %115, label %116, label %118

116:                                              ; preds = %110
  %117 = call i32 @puts(ptr nonnull dereferenceable(1) @str)
  br label %118

118:                                              ; preds = %116, %100, %110, %58, %48
  %119 = phi i32 [ 3, %48 ], [ 4, %58 ], [ 0, %116 ], [ 5, %100 ], [ 6, %110 ]
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %4) #5
  br label %120

120:                                              ; preds = %118, %23, %9
  %121 = phi i32 [ 1, %9 ], [ %119, %118 ], [ 2, %23 ]
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %3) #5
  ret i32 %121
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

declare i64 @__vm_host_call(i32 noundef, i32 noundef, i64 noundef, i64 noundef, i64 noundef, i64 noundef) local_unnamed_addr #2

declare i32 @__vm_cap_resolve(ptr noundef, i64 noundef) local_unnamed_addr #2

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #3

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr #4

attributes #0 = { nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #4 = { nofree nounwind }
attributes #5 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"Ubuntu clang version 18.1.3 (1ubuntu1)"}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"long", !7, i64 0}
