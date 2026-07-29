; ModuleID = '/tmp/claude-0/-home-user-vm/8192c67a-39d3-5f8d-8db5-366ce6cb9abb/scratchpad/pp.bc'
source_filename = "/home/user/vm/crates/svm-llvm/tests/fixtures/posix_probe.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@.str = private unnamed_addr constant [6 x i8] c"posix\00", align 1
@px = internal unnamed_addr global i32 0, align 4
@.str.1 = private unnamed_addr constant [5 x i8] c"ping\00", align 1
@.str.2 = private unnamed_addr constant [3 x i8] c"up\00", align 1
@str = private unnamed_addr constant [15 x i8] c"posix probe ok\00", align 1

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = alloca [2 x i32], align 4
  %2 = alloca [8 x i8], align 4
  %3 = alloca i32, align 4
  %4 = tail call i32 @__vm_cap_resolve(ptr noundef nonnull @.str, i64 noundef 5) #4
  store i32 %4, ptr @px, align 4, !tbaa !5
  %5 = icmp slt i32 %4, 0
  br i1 %5, label %54, label %6

6:                                                ; preds = %0
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %1) #4
  %7 = ptrtoint ptr %1 to i64
  %8 = call i64 @__vm_host_call(i32 noundef %4, i32 noundef 23, i64 noundef %7, i64 noundef 0, i64 noundef 0, i64 noundef 0) #4
  %9 = icmp eq i64 %8, 0
  br i1 %9, label %10, label %52

10:                                               ; preds = %6
  %11 = load i32, ptr %1, align 4, !tbaa !5
  %12 = getelementptr inbounds [2 x i32], ptr %1, i64 0, i64 1
  %13 = load i32, ptr %12, align 4, !tbaa !5
  %14 = sext i32 %13 to i64
  %15 = load i32, ptr @px, align 4, !tbaa !5
  %16 = call i64 @__vm_host_call(i32 noundef %15, i32 noundef 24, i64 noundef %14, i64 noundef 9, i64 noundef 0, i64 noundef 0) #4
  %17 = icmp eq i64 %16, 9
  br i1 %17, label %18, label %52

18:                                               ; preds = %10
  %19 = load i32, ptr @px, align 4, !tbaa !5
  %20 = call i64 @__vm_host_call(i32 noundef %19, i32 noundef 0, i64 noundef 9, i64 noundef ptrtoint (ptr @.str.1 to i64), i64 noundef 4, i64 noundef 0) #4
  %21 = icmp eq i64 %20, 4
  br i1 %21, label %22, label %52

22:                                               ; preds = %18
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %2) #4
  %23 = sext i32 %11 to i64
  %24 = ptrtoint ptr %2 to i64
  %25 = load i32, ptr @px, align 4, !tbaa !5
  %26 = call i64 @__vm_host_call(i32 noundef %25, i32 noundef 1, i64 noundef %23, i64 noundef %24, i64 noundef 8, i64 noundef 0) #4
  %27 = icmp eq i64 %26, 4
  br i1 %27, label %28, label %50

28:                                               ; preds = %22
  %29 = load <4 x i8>, ptr %2, align 4
  %30 = freeze <4 x i8> %29
  %31 = bitcast <4 x i8> %30 to i32
  %32 = icmp eq i32 %31, 1735289200
  br i1 %32, label %33, label %50

33:                                               ; preds = %28
  %34 = load i32, ptr @px, align 4, !tbaa !5
  %35 = call i64 @__vm_host_call(i32 noundef %34, i32 noundef 27, i64 noundef ptrtoint (ptr @.str.2 to i64), i64 noundef 2, i64 noundef 0, i64 noundef 0) #4
  %36 = icmp slt i64 %35, 0
  br i1 %36, label %50, label %37

37:                                               ; preds = %33
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %3) #4
  store i32 0, ptr %3, align 4, !tbaa !5
  %38 = ptrtoint ptr %3 to i64
  %39 = load i32, ptr @px, align 4, !tbaa !5
  %40 = call i64 @__vm_host_call(i32 noundef %39, i32 noundef 28, i64 noundef %35, i64 noundef %38, i64 noundef 0, i64 noundef 0) #4
  %41 = icmp eq i64 %40, %35
  br i1 %41, label %42, label %48

42:                                               ; preds = %37
  %43 = load i32, ptr %3, align 4, !tbaa !5
  %44 = and i32 %43, 65280
  %45 = icmp eq i32 %44, 10752
  br i1 %45, label %46, label %48

46:                                               ; preds = %42
  %47 = call i32 @puts(ptr nonnull dereferenceable(1) @str)
  br label %48

48:                                               ; preds = %42, %37, %46
  %49 = phi i32 [ 0, %46 ], [ 8, %37 ], [ 9, %42 ]
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %3) #4
  br label %50

50:                                               ; preds = %48, %33, %28, %22
  %51 = phi i32 [ 5, %22 ], [ 6, %28 ], [ %49, %48 ], [ 7, %33 ]
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %2) #4
  br label %52

52:                                               ; preds = %50, %10, %18, %6
  %53 = phi i32 [ 2, %6 ], [ %51, %50 ], [ 3, %10 ], [ 4, %18 ]
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %1) #4
  br label %54

54:                                               ; preds = %0, %52
  %55 = phi i32 [ %53, %52 ], [ 1, %0 ]
  ret i32 %55
}

declare i32 @__vm_cap_resolve(ptr noundef, i64 noundef) local_unnamed_addr #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #2

declare i64 @__vm_host_call(i32 noundef, i32 noundef, i64 noundef, i64 noundef, i64 noundef, i64 noundef) local_unnamed_addr #1

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr #3

attributes #0 = { nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nofree nounwind }
attributes #4 = { nounwind }

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
